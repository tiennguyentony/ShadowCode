#!/usr/bin/env bash
# The stop-the-pod command: run train.py on the pod, and when it exits, for
# any reason, keep the result on the volume and stop the pod. Start it in tmux
# (RUNPOD.md):
#
#     bash pod_run.sh <run name> <config> --price-per-hour <price> [more train.py options and overrides]
#     bash pod_run.sh --bench <run name> <config> --max-minutes 10 --price-per-hour <price> [...]
#
# For example, the generated-label run and its 10-minute benchmark:
#
#     bash pod_run.sh labels_generated configs/runpod_labels_generated.yaml --price-per-hour 0.00 data.num_workers=8
#     bash pod_run.sh --bench labels_generated_bench configs/runpod_labels_generated.yaml --max-minutes 10 --price-per-hour 0.00 data.num_workers=8
#
# Everything of the run goes to <VOLUME>/runs/<run name>/: logs/, checkpoints/
# (passed to train.py as log.dir and checkpoint.dir, whatever the config
# says), console.log, kept/. In order:
#
#  1. Checks the arguments and the config. Only a mistake found here ends the
#     script without stopping the pod; you are still watching at that point.
#  2. Starts watchdog.sh for this run in the background: it stops the pod if
#     progress.csv has had no new row for an hour (a hung run never exits).
#  3. Runs train.py, its output to the screen and to console.log, then writes
#     "train.py exited with code N" there (notify.py reads that line).
#  4. With --bench: prints the benchmark (bench_report.py) and saves it as
#     bench_report.txt.
#  5. Runs finish_run.sh: exports the bundle and puts everything worth keeping
#     into kept/<bundle id>/ with checksums (RUNPOD.md, "Keep the result").
#  6. Waits WAIT_BEFORE_STOP seconds (180), so the laptop's notify.py can read
#     why the run ended before the pod goes away.
#  7. Asks RunPod to stop the pod, again every STOP_RETRY seconds (60) until
#     that works. Nothing in steps 3 to 6 can prevent this.
#
# To keep the pod after all, close the tmux session during the wait of step 6
# (tmux kill-session -t <session>). Ctrl-c stops train.py with a checkpoint;
# the steps after it still run and the pod is stopped.
#
# Settings, as environment variables: VOLUME (/workspace), PYTHON (python3),
# WAIT_BEFORE_STOP (180), STOP_RETRY (60), and the watchdog's STALL_SECONDS
# (3600) and CHECK_EVERY (300).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2

VOLUME="${VOLUME:-/workspace}"
PYTHON="${PYTHON:-python3}"
WAIT_BEFORE_STOP="${WAIT_BEFORE_STOP:-180}"
STOP_RETRY="${STOP_RETRY:-60}"

usage() {
    printf 'pod_run.sh: %s\n' "$1" >&2
    echo "usage: bash pod_run.sh [--bench] <run name> <config> --price-per-hour <price> [train.py options and overrides]" >&2
    echo "Nothing was started and the pod was not stopped." >&2
    exit 2
}

bench=0
if [ "${1:-}" = "--bench" ]; then
    bench=1
    shift
fi
NAME="${1:-}"
CONFIG="${2:-}"
[ -n "$NAME" ] && [ -n "$CONFIG" ] || usage "a run name and a config are required"
shift 2
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || usage "the run name '$NAME' may only hold letters, digits, '_', '.' and '-'"
[ -f "$CONFIG" ] || usage "there is no config $CONFIG (run this from the training folder's configs, e.g. configs/runpod_labels_generated.yaml)"

price=""
minutes=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --price-per-hour) price="${args[i + 1]:-}" ;;
        --price-per-hour=*) price="${args[i]#*=}" ;;
        --max-minutes) minutes="${args[i + 1]:-}" ;;
        --max-minutes=*) minutes="${args[i]#*=}" ;;
    esac
done
[ -n "$price" ] || usage "--price-per-hour is required: the pod's price per hour, from the console"
awk -v p="$price" 'BEGIN { exit !(p + 0 > 0) }' || usage "--price-per-hour must be the pod's real price, above 0 (got '$price')"
if [ "$bench" = 1 ]; then
    [ -n "$minutes" ] || usage "--bench needs --max-minutes (10 for the benchmark)"
fi

RUN="$VOLUME/runs/$NAME"
DIRS=("log.dir=$RUN/logs" "checkpoint.dir=$RUN/checkpoints")
overrides=()
for a in "$@"; do
    case "$a" in
        -*) ;;
        *=*) overrides+=("$a") ;;
    esac
done
"$PYTHON" - "$CONFIG" "${overrides[@]}" "${DIRS[@]}" <<'PY' || usage "the config does not load with these overrides (see above)"
import sys
from config import load_config
cfg = load_config(sys.argv[1], sys.argv[2:])
if cfg["data"].get("num_workers") is None:
    sys.exit(f"data.num_workers is not set in {sys.argv[1]}: add data.num_workers=N (setup_runpod.sh prints the fastest N)")
print(f"config {sys.argv[1]}: {cfg['train']['total_steps']} steps, batch {cfg['train']['batch_size']}, "
      f"{cfg['data']['num_workers']} loader workers, data from {cfg['data'].get('source', 'folder')}")
PY

mkdir -p "$RUN/logs" || usage "cannot create $RUN"
LOG="$RUN/console.log"
say() { echo "$*" | tee -a "$LOG"; }
export TORCH_HOME="${TORCH_HOME:-$VOLUME/cache/torch}"
# Ctrl-c reaches train.py, which saves a checkpoint and exits; this script carries on.
trap 'say "pod_run.sh: interrupt received; the pod will still be stopped (close the tmux session to keep it)"' INT

say "launched pod_run.sh $([ "$bench" = 1 ] && echo "--bench ")$NAME at $(date -u '+%Y-%m-%d %H:%M:%S UTC'), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
say "  run folder $RUN; the pod is stopped when train.py exits"

bash watchdog.sh "$NAME" &
watchdog=$!

sampler=""
if [ "$bench" = 1 ]; then
    rm -f "$RUN/gpu_util.csv" "$RUN/cpu_start.txt" "$RUN/cpu_end.txt" "$RUN/logs/throughput.json"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 > "$RUN/gpu_util.csv" 2>/dev/null &
        sampler=$!
    fi
    [ -r /proc/stat ] && head -n 1 /proc/stat > "$RUN/cpu_start.txt"
    nproc > "$RUN/nproc.txt" 2>/dev/null
fi

"$PYTHON" train.py --config "$CONFIG" "$@" "${DIRS[@]}" 2>&1 | tee -ia "$LOG"
code=${PIPESTATUS[0]}
say "train.py exited with code $code at $(date -u)"

if [ "$bench" = 1 ]; then
    [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
    [ -r /proc/stat ] && head -n 1 /proc/stat > "$RUN/cpu_end.txt"
    "$PYTHON" bench_report.py "$RUN" --labelgen "$VOLUME/cache/labelgen_bench.txt" > "$RUN/bench_report.txt" 2>&1 \
        || echo "bench_report.py failed (above)" >> "$RUN/bench_report.txt"
    tee -a "$LOG" < "$RUN/bench_report.txt"
fi

if [ -f finish_run.sh ]; then
    bash finish_run.sh "$RUN" "$PWD" 2>&1 | tee -a "$LOG"
else
    say "finish_run.sh is not in $PWD; nothing was staged"
fi

[ -n "${RUNPOD_POD_ID:-}" ] || say "WARNING: RUNPOD_POD_ID is not set, so the stop below cannot work. Stop the pod in the RunPod console."
command -v runpodctl >/dev/null || say "WARNING: runpodctl is not installed, so the stop below cannot work. Stop the pod in the RunPod console."
say "stopping the pod in $WAIT_BEFORE_STOP s (close the tmux session now to keep it)"
sleep "$WAIT_BEFORE_STOP"
kill "$watchdog" 2>/dev/null
# The second form is the older runpodctl syntax, for images that ship an older version.
try_stop() {
    runpodctl pod stop "${RUNPOD_POD_ID:-}" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] && return 0
    runpodctl stop pod "${RUNPOD_POD_ID:-}" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ]
}
until try_stop; do
    say "the pod did not stop at $(date -u); trying again in $STOP_RETRY s"
    sleep "$STOP_RETRY"
done
say "runpodctl accepted the stop at $(date -u)"
