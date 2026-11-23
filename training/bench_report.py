"""The 10-minute benchmark's report, from what pod_run.sh --bench left in the run folder.

    python bench_report.py /workspace/runs/labels_generated_bench [--labelgen /workspace/cache/labelgen_bench.txt]

pod_run.sh --bench runs this by itself after train.py exits. It reads:

    logs/throughput.json   train.py: steps, seconds, time spent waiting for the data loader
    logs/progress.csv      train.py: one row per 1,000 steps, with hours and the image-loss weight
    gpu_util.csv           nvidia-smi every 5 s during the run: utilisation %, memory MiB
    cpu_start.txt, cpu_end.txt, nproc.txt   the first line of /proc/stat before and after, nproc
    --labelgen             setup_runpod.sh's labelgen.py --bench output (tiles per second)

and prints steps per second, GPU utilisation, whether the CPU (label
generation in the loader processes) held the GPU up, and the hours and GPU
cost of the whole run at that speed. Every number is measured on this pod in
this benchmark; a file that is missing is reported as missing, not guessed.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

# The CPU is called the bottleneck when the training loop spent at least this
# share of its time waiting for the next batch ...
WAIT_BOTTLENECK = 0.10
# ... and "possibly" when it waited at least this share and the GPU was busy less than GPU_BUSY percent.
WAIT_POSSIBLE = 0.03
GPU_BUSY = 80.0
BENCH_LINE = re.compile(r"^(\d+) worker\(s\): .* = ([0-9.]+) tiles/s")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def progress_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("step", "").isdigit() and r.get("elapsed_hours")]


def gpu_samples(path: Path) -> tuple[list[float], list[float]]:
    util, memory = [], []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [p.strip() for p in line.split(",")]
            try:
                util.append(float(parts[0]))
                memory.append(float(parts[1]))
            except (IndexError, ValueError):
                continue
    return util, memory


def cpu_busy(start: Path, end: Path) -> float | None:
    """Share of all CPU time that was busy between the two /proc/stat lines."""
    try:
        a = [int(x) for x in start.read_text().split()[1:]]
        b = [int(x) for x in end.read_text().split()[1:]]
    except (OSError, ValueError):
        return None
    delta = [y - x for x, y in zip(a, b)]
    total = sum(delta[:8])  # user nice system idle iowait irq softirq steal; guest is already in user
    idle = delta[3] + (delta[4] if len(delta) > 4 else 0)
    return None if total <= 0 else 1.0 - idle / total


def labelgen_bench(path: Path | None) -> dict[int, float]:
    if path is None or not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BENCH_LINE.match(line.strip())
        if m:
            found[int(m.group(1))] = float(m.group(2))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="the run folder pod_run.sh --bench wrote")
    parser.add_argument("--labelgen", help="setup_runpod.sh's labelgen benchmark output")
    parser.add_argument("--steps", type=int, help="steps of the whole run (default: train.total_steps of the benchmark)")
    args = parser.parse_args()
    run = Path(args.run)

    t = read_json(run / "logs" / "throughput.json")
    print("\n== benchmark report")
    if t is None:
        print(f"  no {run / 'logs' / 'throughput.json'}: train.py did not reach the end of its loop (a crash?).")
        print("  Nothing to report; console.log has train.py's output.")
        return 1
    steps, seconds = int(t["steps"]), float(t["seconds"])
    if steps <= 0 or seconds <= 0:
        print("  train.py trained no steps in this process; nothing to report.")
        return 1
    total = args.steps or int(t["total_steps"])
    batch, workers = int(t["batch_size"]), int(t["num_workers"])
    rate = steps / seconds
    print(f"  GPU            {t.get('device')}, {workers} loader workers, batch {batch}")
    print(f"  steps/s        {rate:.2f} over this process: steps {t['start_step']} to {t['end_step']} in {seconds / 60:.1f} min,")
    print("                 sample sheets, spread checks and checkpoints included, as in the real run")

    # The image loss (L2, LPIPS, peak penalty) only runs once bit accuracy passes 0.95; until then steps are cheaper.
    rows = progress_rows(run / "logs" / "progress.csv")
    on_rates = []
    for before, after in zip(rows, rows[1:]):
        hours = float(after["elapsed_hours"]) - float(before["elapsed_hours"])
        if float(after.get("image_scale") or 0) > 0 and hours > 0:
            on_rates.append((int(after["step"]) - int(before["step"])) / (hours * 3600.0))
    if on_rates:
        on_rate = statistics.median(on_rates)
        print(f"  steps/s        {on_rate:.2f} with the image loss on (median of {len(on_rates)} intervals of 1,000 steps)")
    elif t.get("image_loss_on_at_end"):
        on_rate = None
        print("  image loss     came on near the end: too few steps with it on to time separately")
    else:
        on_rate = None
        print("  image loss     off for the whole benchmark. Once it is on, LPIPS and the peak penalty add work to")
        print("                 every step, so the real run's later steps are slower than this; by how much is not measured.")

    util, memory = gpu_samples(run / "gpu_util.csv")
    if util:
        print(f"  GPU busy       mean {statistics.mean(util):.0f} %, median {statistics.median(util):.0f} % "
              f"({len(util)} nvidia-smi samples, 5 s apart; includes start-up), memory up to {max(memory):.0f} MiB")
    else:
        print("  GPU busy       not measured (no gpu_util.csv: nvidia-smi missing?)")

    first = float(t.get("first_batch_seconds") or 0.0)
    wait = float(t["loader_wait_seconds"])
    after_first = max(1e-9, seconds - first)
    share = wait / after_first
    print(f"  data loader    the loop waited {wait:.1f} s for batches: {share * 100:.1f} % of the time after the first batch"
          f" (the first took {first:.1f} s: starting the workers)")
    need = rate * batch
    vcpus = os.environ.get("RUNPOD_CPU_COUNT") or _read(run / "nproc.txt")
    busy = cpu_busy(run / "cpu_start.txt", run / "cpu_end.txt")
    print(f"  CPU            {vcpus or '?'} vCPUs; " + (f"{busy * 100:.0f} % busy over the run (all CPUs)" if busy is not None else "load not measured"))
    bench = labelgen_bench(Path(args.labelgen) if args.labelgen else None)
    if bench:
        best = max(bench, key=bench.get)
        at_workers = f", {bench[workers]:.1f} with the {workers} this run used" if workers in bench and workers != best else ""
        print(f"  tiles/s        the run used {need:.1f}; labelgen.py --bench made up to {bench[best]:.1f} with {best} workers{at_workers}")
    else:
        print(f"  tiles/s        the run used {need:.1f}; no labelgen benchmark found to compare with")

    gpu_mean = statistics.mean(util) if util else None
    if share >= WAIT_BOTTLENECK:
        verdict = (f"YES: the GPU waited for tiles {share * 100:.0f} % of the time. More loader workers (if the labelgen "
                   "benchmark scales) or a pod with more vCPUs would make the run faster.")
    elif share >= WAIT_POSSIBLE and gpu_mean is not None and gpu_mean < GPU_BUSY:
        verdict = (f"POSSIBLY: the loop waited {share * 100:.0f} % of the time and the GPU was busy {gpu_mean:.0f} % on average.")
    else:
        verdict = f"NO: the loop waited for tiles {share * 100:.1f} % of the time."
        if gpu_mean is not None and gpu_mean < GPU_BUSY:
            verdict += (f" The GPU was busy only {gpu_mean:.0f} %, so something other than the loader limits the speed"
                        " (the training loop itself, sample sheets, small kernels).")
    print(f"  CPU bottleneck {verdict}")
    print(f"                 (yes at {WAIT_BOTTLENECK * 100:.0f} % or more waiting; possibly at {WAIT_POSSIBLE * 100:.0f} % "
          f"with the GPU under {GPU_BUSY:.0f} % busy)")

    use = on_rate if on_rate is not None else rate
    hours = total / use / 3600.0
    price = t.get("price_per_hour")
    basis = "the image-loss-on speed" if on_rate is not None else "this benchmark's speed, image loss off (an underestimate)"
    line = f"  estimate       {total:,} steps at {use:.2f} steps/s ({basis}): {hours:.1f} h"
    if price:
        line += f", ${hours * float(price):.2f} at ${float(price):g} per hour"
    print(line)
    print("                 train.py's time only: setup, the export, the 3-minute wait before the stop, idle time")
    print("                 and storage are not in it. The RunPod billing page is the real figure.")
    limit = t.get("max_hours")
    if limit is not None and hours > float(limit):
        print(f"                 Over {float(limit):g} h: budget.max_hours stops the run then, near step "
              f"{int(use * float(limit) * 3600):,} at this speed.")
    return 0


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())
