#!/usr/bin/env bash
# Prepare a RunPod pod for training. Run it from this folder after every pod
# start (a new container has lost its pip packages):
#
#     bash setup_runpod.sh
#
# It installs the dependencies, checks the GPU, and fetches the dataset to the
# network volume if it is not there yet. It is safe to run again. It does not
# start training and it does not start or stop anything that is billed.
#
# Settings, as environment variables:
#     CONFIG          config whose data.train_dir is filled   (configs/runpod_spread.yaml)
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
    echo "  pod id $RUNPOD_POD_ID, runpodctl found: the stop-when-done command in RUNPOD.md will work"
else
    echo "  RUNPOD_POD_ID or runpodctl is missing: stop the pod from the RunPod console instead (RUNPOD.md)"
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
    print(f"  below {early['min_bit_acc']}. The pod keeps billing after that unless you use the")
    print("  stop-when-done command or stop it yourself.")
if spread.get("stop_above") is not None:
    print(f"  It also stops if the mark piles up at the same place in every tile: fixed-place measure")
    print(f"  above {spread['stop_above']:g} at a sample sheet.")
image_check = cfg.get("image_loss_check") or {}
if image_check.get("enabled"):
    print(f"  It also stops if PSNR is still below {image_check['min_psnr_db']:g} dB at step {image_check['at_step']}:")
    print("  the image loss never switched on.")
PY
cat <<EOF

  To train, in tmux (RUNPOD.md, "Run"):

      export TORCH_HOME=$TORCH_HOME
      $PYTHON train.py --config $CONFIG --price-per-hour <your pod's price>

EOF
