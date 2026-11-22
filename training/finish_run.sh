#!/usr/bin/env bash
# Put everything that has to be kept from a run into one folder on the volume,
# with a checksum for every file. For the pod, when the run has ended:
#
#     bash finish_run.sh
#
# keep_result.py on the laptop runs this over ssh by itself, then copies the
# folder and checks the copy (RUNPOD.md, "Keep the result"). It writes
#
#     <run>/kept/<bundle id>/
#         checkpoint/step_XXXXXXXX.pt   the newest checkpoint: models, optimizer, schedule state
#         bundle/                       export_bundle.py's output: reader.onnx, contract.json, fixtures/
#         config/                       the config files, and resolved.json: the config stored in the checkpoint
#         COMMIT                        the commit of the code on the pod, and any local changes
#         run/                          progress.csv, spread.csv, run_state.json, console.log, last sample sheets
#         environment.txt               python, torch and package versions, the GPU
#         SHA256SUMS                    a checksum for every file above
#
# The bundle id is export_bundle.py's: the date and the short commit.
#
# Safe to run again: a checkpoint that is already staged is not staged twice.
# It changes nothing outside <run>/kept and <run>/export, and stops nothing.
#
# Arguments, or environment variables, all optional:
#     $1 / RUN      the run folder                  (/workspace/runs/spread)
#     $2 / CODE     the training folder of the code  (/workspace/ShadowCode/training)
#     PYTHON        (python3)
set -uo pipefail
RUN="${1:-${RUN:-/workspace/runs/spread}}"
CODE="${2:-${CODE:-/workspace/ShadowCode/training}}"
PYTHON="${PYTHON:-python3}"

fail() { printf 'FAILED: %s\n' "$1"; exit 1; }

newest="$(ls "$RUN"/checkpoints/step_*.pt 2>/dev/null | sort | tail -n 1)"
[ -n "$newest" ] || fail "no checkpoint in $RUN/checkpoints"
cd "$CODE" || fail "the code folder $CODE does not exist"
echo "newest checkpoint: $newest ($(du -h "$newest" | cut -f1))"
sum="$(sha256sum "$newest" | cut -d' ' -f1)"

staged="$(grep -l "^$sum" "$RUN"/kept/*/SHA256SUMS 2>/dev/null | head -n 1)"
if [ -n "$staged" ]; then
    KEPT="$(dirname "$staged")"
    echo "already staged in $KEPT"
else
    commit="$(git rev-parse HEAD 2>&1)" || fail "cannot read the git commit in $CODE: $commit"
    id="$(date +%Y%m%d)-$(git rev-parse --short HEAD)"
    export_log="$(mktemp)"
    echo "exporting the bundle (ONNX reader, contract.json, fixtures) ..."
    if "$PYTHON" export_bundle.py "$newest" --out "$RUN/export" > "$export_log" 2>&1; then
        exported=yes
        id="$(sed -n 's/^bundle_id \([^ ]*\).*/\1/p' "$export_log" | tail -n 1)"
    else
        exported=no
        echo "the export failed; the checkpoint is staged without a bundle. Last lines:"
        tail -n 5 "$export_log" | sed 's/^/    /'
    fi
    KEPT="$RUN/kept/$id"
    part="$KEPT.partial"
    rm -rf "$part"
    mkdir -p "$part/checkpoint" "$part/config" "$part/run" || fail "cannot write to $RUN/kept"

    cp "$newest" "$part/checkpoint/" || fail "could not copy the checkpoint (is the volume full?)"
    if [ "$exported" = yes ]; then
        cp -r "$RUN/export/$id" "$part/bundle" || fail "could not copy the bundle"
    else
        cp "$export_log" "$part/EXPORT_FAILED.txt"
    fi
    cp "$export_log" "$part/run/export.log"

    cp configs/*.yaml "$part/config/"
    "$PYTHON" - "$newest" "$part/config/resolved.json" <<'PY' || echo "note: could not write config/resolved.json"
import json, sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
json.dump({"step": int(state["step"]), "image_ramp_start": state["image_ramp_start"], "acc_ema": float(state["acc_ema"]),
           "band": state["band"], "config": state["config"]}, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
PY

    {
        echo "commit $commit"
        echo "branch $(git rev-parse --abbrev-ref HEAD)"
        echo "remote $(git remote get-url origin 2>/dev/null)"
        echo "staged $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "checkpoint $(basename "$newest") sha256 $sum"
        echo
        echo "This is the commit checked out when the result was staged. train.py loaded its"
        echo "code when the run started; if the checkout was changed during the run, this is"
        echo "not the code that trained. Local changes in /training (git status), none if empty:"
        git status --porcelain -- .
    } > "$part/COMMIT"

    for f in logs/progress.csv logs/spread.csv logs/run_state.json console.log; do
        [ -f "$RUN/$f" ] && cp "$RUN/$f" "$part/run/"
    done
    mkdir -p "$part/run/samples"
    for f in $(ls "$RUN"/logs/samples/step_*.png 2>/dev/null | sort | tail -n 3) $(ls "$RUN"/logs/samples/final_*.png 2>/dev/null); do
        cp "$f" "$part/run/samples/"
    done

    {
        "$PYTHON" -V
        "$PYTHON" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
        echo
        "$PYTHON" -m pip freeze 2>/dev/null
    } > "$part/environment.txt" 2>&1

    (cd "$part" && find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | tr '\n' '\0' | xargs -0 sha256sum > SHA256SUMS) \
        || fail "could not write the checksums"
    if [ -e "$KEPT" ]; then
        mv "$KEPT" "$KEPT.replaced-$(date +%H%M%S)"
        echo "an earlier result with the same id was moved aside, not deleted"
    fi
    mv "$part" "$KEPT" || fail "could not move the staged folder into place"
    rm -f "$export_log"
fi

echo
echo "staged: $KEPT ($(du -sh "$KEPT" | cut -f1))"
# The lines keep_result.py reads.
echo "KEPT_DIR $KEPT"
echo "KEPT_ID $(basename "$KEPT")"
echo "KEPT_FILES $(wc -l < "$KEPT/SHA256SUMS" | tr -d ' ')"
echo "KEPT_SUMS $(sha256sum "$KEPT/SHA256SUMS" | cut -d' ' -f1)"
echo "KEPT_BUNDLE $([ -f "$KEPT/bundle/contract.json" ] && echo yes || echo no)"
