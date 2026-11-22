"""Copy a finished run's result from the pod to the laptop and check the copy.

    python keep_result.py --pod root@<pod ip> --port <port> --key ~/.ssh/<key>
    python keep_result.py --check kept/<bundle id>        check a copy again; no pod needed

Three steps, from the laptop:

1. Stage. finish_run.sh is run on the pod over ssh (it is sent with the
   command; nothing has to be installed there). It exports the bundle from
   the newest checkpoint and puts checkpoint, bundle, config, commit and run
   logs into <run>/kept/<bundle id>/ with a checksum for every file.
2. Copy. That folder is copied with scp to kept/<bundle id>/ here.
3. Check. Every file here is hashed again and compared with the pod's
   checksums; nothing may be missing or extra, and the checksum list itself
   must be the one the pod has.

Only when it prints CHECK PASSED may the pod and the volume be deleted. The
exit code says the same: 0 passed, 1 failed, 2 the copy is good but holds no
exported bundle (the export failed on the pod; the message says what to do).

It needs the pod running with full SSH (public IP, TCP port 22). The pod is
not stopped and nothing on it is deleted. RUNPOD.md, "Keep the result".

Standard library only. It never imports torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from podlink import last_line, load_settings, pod_from

HERE = Path(__file__).resolve().parent
NOT_IN_LIST = {"SHA256SUMS", "VERIFIED.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    """SHA256SUMS as {relative path: checksum}. sha256sum writes "<hash>  <path>" or "<hash> *<path>"."""
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition(" ")
            sums[name.strip().lstrip("*")] = digest
    return sums


def check(folder: Path, pod_sums_checksum: str | None) -> int:
    """Compare ``folder`` with its SHA256SUMS. Returns the exit code and prints the verdict."""
    problems: list[str] = []
    listing = folder / "SHA256SUMS"
    if not listing.is_file():
        print(f"CHECK FAILED: {folder} has no SHA256SUMS. Do not delete the pod or the volume.")
        return 1
    listing_sum = sha256(listing)
    if pod_sums_checksum is None:
        print("note: the pod was not asked, so the checksum list itself is taken as it is here")
    elif listing_sum != pod_sums_checksum:
        problems.append(f"SHA256SUMS differs from the pod's (here {listing_sum[:12]}, pod {pod_sums_checksum[:12]})")

    expected = read_sums(listing)
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - NOT_IN_LIST
    for name in sorted(expected.keys() - present):
        problems.append(f"missing: {name}")
    for name in sorted(present - expected.keys()):
        problems.append(f"not in the pod's list: {name}")
    total = 0
    for name in sorted(expected.keys() & present):
        path = folder / name
        total += path.stat().st_size
        if sha256(path) != expected[name]:
            problems.append(f"checksum differs: {name}")

    # What a result has to hold, beyond matching the pod.
    checkpoints = sorted(n for n in expected if n.startswith("checkpoint/") and n.endswith(".pt"))
    if len(checkpoints) != 1:
        problems.append(f"expected one checkpoint, the list names {len(checkpoints)}")
    for name in ("config/resolved.json", "COMMIT"):
        if name not in expected:
            problems.append(f"the list has no {name}")
    has_bundle = "bundle/contract.json" in expected and "bundle/reader.onnx" in expected
    if has_bundle and not problems:
        contract = json.loads((folder / "bundle" / "contract.json").read_text(encoding="utf-8"))
        if contract["models"]["reader"]["sha256"] != expected["bundle/reader.onnx"]:
            problems.append("contract.json names another reader.onnx than the one in the bundle")
        step = int(checkpoints[0].rsplit("_", 1)[-1].removesuffix(".pt"))
        if not contract.get("trained") or contract.get("checkpoint_step") != step:
            problems.append(
                f"the bundle was not exported from this checkpoint (bundle: trained {contract.get('trained')}, "
                f"step {contract.get('checkpoint_step')}; checkpoint: step {step})"
            )

    if problems:
        print(f"\nCHECK FAILED for {folder}:")
        for problem in problems:
            print(f"  {problem}")
        print("Do not delete the pod or the volume. Run the command again; it copies afresh.")
        (folder / "VERIFIED.txt").unlink(missing_ok=True)
        return 1

    commit = (folder / "COMMIT").read_text(encoding="utf-8").splitlines()[0]
    lines = [
        f"checked {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"files {len(expected)}, {total:,} bytes, every checksum equal to the pod's list",
        f"SHA256SUMS sha256 {listing_sum}" + ("" if pod_sums_checksum is None else " (equal to the pod's)"),
        f"checkpoint {checkpoints[0]} sha256 {expected[checkpoints[0]]}",
        commit,
        f"bundle {'present, exported from this checkpoint' if has_bundle else 'MISSING: the export failed on the pod'}",
    ]
    (folder / "VERIFIED.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    for line in lines:
        print(f"  {line}")
    if not has_bundle:
        print(
            f"\nCOPY GOOD, BUNDLE MISSING: {folder}\n"
            "The checkpoint, config, commit and logs are here and match the pod. The export failed on the pod\n"
            f"({folder / 'EXPORT_FAILED.txt'} has its output). Export here, on the CPU, before deleting anything:\n"
            f"    python export_bundle.py {(folder / checkpoints[0]).as_posix()} --out {(folder / 'bundle-from-laptop').as_posix()}\n"
            "Hide the GPU first if a training run is using it (PowerShell: $env:CUDA_VISIBLE_DEVICES = '-1')."
        )
        return 2
    print(f"\nCHECK PASSED: {folder}\nThe pod and the volume may now be deleted (RUNPOD.md, \"Keep the result\").")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pod", help="root@<pod ip> (default: \"pod\" in ~/.shadowcode/notify.json)")
    parser.add_argument("--port", type=int, help="the pod's public SSH port")
    parser.add_argument("--key", help="private key registered with RunPod, e.g. ~/.ssh/id_ed25519")
    parser.add_argument("--pod-run", default="/workspace/runs/spread", help="run folder on the pod")
    parser.add_argument("--pod-code", default="/workspace/ShadowCode/training", help="training folder of the code on the pod")
    parser.add_argument("--dest", default="kept", help="local folder the result folder is created in (git-ignored)")
    parser.add_argument("--check", metavar="FOLDER", help="only check a result folder that is already here")
    args = parser.parse_args()

    if args.check:
        return check(Path(args.check), None)

    pod = pod_from(args, load_settings())
    if pod is None:
        parser.error("no pod given: pass --pod root@<pod ip> --port <port>")

    print(f"1/3 staging on the pod ({pod.host}, port {pod.port}); the export takes a minute or two")
    script = (HERE / "finish_run.sh").read_text(encoding="utf-8")
    try:
        done = pod.run_script(script, args.pod_run, args.pod_code, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"FAILED: could not run the staging on the pod ({type(error).__name__}: {error})")
        return 1
    out = done.stdout.decode("utf-8", errors="replace")
    print("".join(f"    {line}\n" for line in out.splitlines() if not line.startswith("KEPT_")), end="")
    facts = dict(line.split(" ", 1) for line in out.splitlines() if line.startswith("KEPT_") and " " in line)
    if done.returncode != 0 or "KEPT_DIR" not in facts or "KEPT_SUMS" not in facts:
        print(f"FAILED: the staging did not finish on the pod ({last_line(done.stderr) if done.returncode else 'no result line'}).")
        print("Nothing was copied. Do not delete the pod or the volume.")
        return 1

    dest = (HERE / args.dest).resolve()
    target = dest / facts["KEPT_ID"]
    if target.exists():
        aside = target.with_name(f"{target.name}.previous-{time.strftime('%Y%m%d-%H%M%S')}")
        os.replace(target, aside)
        print(f"    an earlier copy was moved to {aside.name}")
    print(f"2/3 copying {facts['KEPT_DIR']} ({facts.get('KEPT_FILES', '?')} files) to {target}")
    try:
        pod.copy_from(facts["KEPT_DIR"], dest, recursive=True)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"FAILED: the copy did not finish ({error}). Do not delete the pod or the volume.")
        return 1

    print("3/3 checking the copy against the pod's checksums")
    return check(target, facts["KEPT_SUMS"])


if __name__ == "__main__":
    sys.exit(main())
