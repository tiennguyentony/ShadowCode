#!/usr/bin/env bash
# Prepare a RunPod pod for training. Run it from this folder after every pod
# start (a new container has lost its pip packages):
#
#     bash setup_runpod.sh
#     CONFIG=configs/runpod_labels_generated.yaml bash setup_runpod.sh
#
# It installs the dependencies, checks the GPU, and fetches the dataset to the
# network volume if it is not there yet. It is safe to run again. It does not
# start training and it does not start or stop anything that is billed.
#
# A config with data.source: generated_labels has no dataset to fetch: its
# tiles are drawn on the CPU by labelgen.py while it trains. Instead, the
# script times labelgen.py with 1, 4 and 8 loader processes (and the pod's
# vCPU count, if higher), prints the fastest, and saves the timing to
# <VOLUME>/cache/labelgen_bench.txt for the benchmark report.
#
# Settings, as environment variables:
#     CONFIG          the config to prepare                    (configs/runpod_spread.yaml)
#     VOLUME          where the network volume is mounted      (/workspace)
#     DATASET_COUNT   images to fetch, or "all"                (16000 = 4.94 GB)
#     ALLOW_LARGE=1   permit a dataset download over 5 GB
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-configs/runpod_spread.yaml}"
VOLUME="${VOLUME:-/workspace}"
DATASET_COUNT="${DATASET_COUNT:-16000}"
PYTHON="${PYTHON:-python3}"

step() { printf '\n== %s\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

step "1/4 network volume"
[ -d "$VOLUME" ] || fail "$VOLUME does not exist. Attach the network volume when you deploy the pod (RUNPOD.md)."
touch "$VOLUME/.write_test" 2>/dev/null && rm -f "$VOLUME/.write_test" || fail "$VOLUME is not writable."
df -h "$VOLUME" | sed 's/^/  /'
echo "  Check the line above: it should show the size of your network volume. If it shows the"
echo "  container disk instead, the volume is not attached and nothing written here will survive."

step "2/4 dependencies"
command -v "$PYTHON" >/dev/null || fail "$PYTHON not found. Use a PyTorch pod template (RUNPOD.md)."
"$PYTHON" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ! command -v tmux >/dev/null; then
    echo "  installing tmux"
    (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null || echo "  could not install tmux; install it by hand"
fi
"$PYTHON" - <<'PY'
import lpips, PIL, tensorboard, torch, torchvision, yaml  # noqa: F401
print(f"  python packages ok: torch {torch.__version__}, torchvision {torchvision.__version__}")
PY

step "3/4 GPU"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: this pod has no NVIDIA GPU."
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv | sed 's/^/  /'
"$PYTHON" - <<'PY' || fail "PyTorch cannot use the GPU. Do not start training on this pod."
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("  torch.cuda.is_available() is False")
x = torch.randn(2, 3, 384, 384, device="cuda")
y = torch.nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular").cuda()(x)
torch.cuda.synchronize()
gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"  PyTorch ran a convolution on {torch.cuda.get_device_name(0)} ({gb:.0f} GiB), result {tuple(y.shape)}")
PY

SOURCE="$("$PYTHON" -c "from config import load_config; print(load_config('$CONFIG')['data'].get('source') or 'folder')")"
RUN_NAME="$("$PYTHON" -c "from pathlib import Path; from config import load_config; d = Path(load_config('$CONFIG')['log']['dir']); print((d.parent if d.name == 'logs' else d).name)")"
BEST_WORKERS=""
if [ "$SOURCE" = generated_labels ]; then
    step "4/4 label generator (data.source generated_labels: no dataset to download)"
    VCPUS="${RUNPOD_CPU_COUNT:-$(nproc)}"
    echo "  vCPUs: RUNPOD_CPU_COUNT=${RUNPOD_CPU_COUNT:-unset}, nproc $(nproc)"
    COUNTS=(1 4 8)
    if [ "$VCPUS" -gt 8 ] 2>/dev/null; then COUNTS+=("$VCPUS"); fi
    mkdir -p "$VOLUME/cache"
    BENCH_FILE="$VOLUME/cache/labelgen_bench.txt"
    echo "  timing labelgen.py: 300 tiles per process, with ${COUNTS[*]} processes"
    bench_code=0
    "$PYTHON" labelgen.py --bench 300 --workers "${COUNTS[@]}" > "$BENCH_FILE" 2>&1 || bench_code=$?
    sed 's/^/  /' "$BENCH_FILE"
    [ "$bench_code" -eq 0 ] || fail "labelgen.py --bench failed (above). Do not start training on this pod."
    BEST="$(sed -n 's/^\([0-9]*\) worker(s): .* = \([0-9.]*\) tiles\/s.*/\2 \1/p' "$BENCH_FILE" | sort -g -r | head -n 1)"
    [ -n "$BEST" ] || fail "no timing line in $BENCH_FILE"
    BEST_WORKERS="${BEST#* }"
    echo "  fastest: $BEST_WORKERS workers, ${BEST% *} tiles/s (saved in $BENCH_FILE)"
    echo "  At batch 8 that is enough for $(awk -v t="${BEST% *}" 'BEGIN { printf "%.1f", t / 8 }') steps per second at most, if training"
    echo "  left the CPU to the loader. The 10-minute benchmark (RUNPOD.md) shows whether the GPU waits."
else
    step "4/4 dataset"
    TRAIN_DIR="$("$PYTHON" -c "from config import load_config; print(load_config('$CONFIG')['data']['train_dir'])")"
    case "$TRAIN_DIR" in
        "$VOLUME"/*/images) ;;
        *) fail "data.train_dir in $CONFIG is $TRAIN_DIR; expected a path under $VOLUME ending in /images." ;;
    esac
    DATASET_DIR="$(dirname "$TRAIN_DIR")"
    FETCH_ARGS=(--out "$DATASET_DIR" --count "$DATASET_COUNT")
    [ "${ALLOW_LARGE:-0}" = "1" ] && FETCH_ARGS+=(--allow-large)
    "$PYTHON" fetch_dataset.py "${FETCH_ARGS[@]}" || fail "the dataset is not complete (see above). Run this script again to continue; a download over 5 GB also needs ALLOW_LARGE=1."
    echo "  $(find "$TRAIN_DIR" -name '*.jpg' | wc -l) images in $TRAIN_DIR, $(du -sh "$DATASET_DIR" | cut -f1) in $DATASET_DIR"
fi

# Pretrained weights for the LPIPS loss go to the volume too, so they are downloaded once.
export TORCH_HOME="$VOLUME/cache/torch"
mkdir -p "$TORCH_HOME"
"$PYTHON" - "$CONFIG" <<'PY'
import sys
from config import load_config
cfg = load_config(sys.argv[1])
if cfg["loss"]["lpips_weight"] > 0:
    from losses import PerceptualLoss
    PerceptualLoss(cfg["loss"]["lpips_net"])
    print(f"  LPIPS weights ({cfg['loss']['lpips_net']}) are in place; licence position in THIRD_PARTY.md")
import os
for key in ("checkpoint", "log"):
    os.makedirs(cfg[key]["dir"], exist_ok=True)
print(f"  checkpoints will go to {cfg['checkpoint']['dir']}")
print(f"  logs will go to        {cfg['log']['dir']}")
PY

step "ready - nothing has been started"
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "  pod id $RUNPOD_POD_ID, runpodctl found: pod_run.sh can stop the pod (RUNPOD.md)"
else
    echo "  RUNPOD_POD_ID or runpodctl is missing: pod_run.sh cannot stop the pod; stop it from the RunPod console (RUNPOD.md)"
fi
echo
"$PYTHON" - "$CONFIG" <<'PY'
import sys
from config import load_config
cfg = load_config(sys.argv[1])
if cfg["train"]["reader_input"] == "direct":
    print("  Direct mode has not yet been shown to learn.")
spread = cfg.get("spread") or {}
if any((spread.get(name) or {}).get("enabled") for name in ("peak_penalty", "hide")):
    print("  The spread options in this config have not been used in a training run before.")
early = cfg.get("early_stop") or {}
if early.get("enabled"):
    print(f"  The run stops itself after {early['after_minutes']:g} minutes of training if bit accuracy is")
    print(f"  below {early['min_bit_acc']}. The pod keeps billing after that unless it was started")
    print("  with pod_run.sh or you stop it yourself.")
if spread.get("stop_above") is not None:
    print(f"  It also stops if the mark piles up at the same place in every tile: fixed-place measure")
    print(f"  above {spread['stop_above']:g} at a sample sheet.")
image_check = cfg.get("image_loss_check") or {}
if image_check.get("enabled"):
    print(f"  It also stops if PSNR is still below {image_check['min_psnr_db']:g} dB at step {image_check['at_step']}:")
    print("  the image loss never switched on.")
budget = cfg.get("budget") or {}
limits = [f"{budget['max_hours']:g} hours of training"] if budget.get("max_hours") is not None else []
limits += [f"${budget['max_cost_usd']:g} (needs --price-per-hour)"] if budget.get("max_cost_usd") is not None else []
if limits:
    print(f"  It also stops at {' or '.join(limits)}, summed over restarts.")
else:
    print("  Nothing in this config limits hours or cost: the run goes on to its last step.")
PY
WORKERS_ARG=""
[ -n "$BEST_WORKERS" ] && WORKERS_ARG=" data.num_workers=$BEST_WORKERS"
cat <<EOF

  In tmux (RUNPOD.md), with your pod's price per hour in place of 0.00. Each one stops the pod when it ends.

  The 10-minute benchmark:

      bash pod_run.sh --bench ${RUN_NAME}_bench $CONFIG --max-minutes 10 --price-per-hour 0.00$WORKERS_ARG

  The run:

      bash pod_run.sh $RUN_NAME $CONFIG --price-per-hour 0.00$WORKERS_ARG

EOF
