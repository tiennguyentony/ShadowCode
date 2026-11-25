"""Checkpoints: save, find the latest, resume."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

_PATTERN = re.compile(r"^step_(\d+)\.pt$")


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Checkpoint files in ``directory``, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := _PATTERN.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(directory: str | Path) -> Path | None:
    files = list_checkpoints(directory)
    return files[-1] if files else None


def save_checkpoint(trainer, directory: str | Path, keep: int) -> Path:
    """Write ``step_XXXXXXXX.pt`` and delete all but the newest ``keep`` checkpoints.

    The file is written under a temporary name and renamed, so a run killed
    mid-save never leaves a truncated checkpoint as the latest one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{trainer.step:08d}.pt"
    state = {
        "step": trainer.step,
        "models": {name: module.state_dict() for name, module in trainer.models.named().items()},
        "optimizer": trainer.optimizer.state_dict(),
        "acc_ema": trainer.acc_ema,
        "image_ramp_start": trainer.image_ramp_start,
        # The image-loss weight the next step trains with. With
        # schedule.image_loss.follow_accuracy it is state of its own; without, the step count gives it.
        "image_scale": trainer.image_loss_scale(),
        # schedule.image_loss.follow_visibility: the visibility-loss weight and its smoothed measure (None before the trigger).
        "vis_weight": trainer.vis_weight,
        "vis_ema": trainer.vis_ema,
        "band": trainer.models.band.as_dict(),
        "reader_input": trainer.cfg["train"]["reader_input"],
        "config": trainer.cfg,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    if keep > 0:
        for old in list_checkpoints(directory)[:-keep]:
            old.unlink()
    return path


def set_aside_checkpoints(directory: str | Path, label: str) -> Path | None:
    """Move every checkpoint in ``directory`` into ``directory/replaced-<label>``.

    Used by a fresh start. Left in place, the old files would have higher step
    numbers than the new run's, so they would be the ones kept and resumed from.
    """
    files = list_checkpoints(directory)
    if not files:
        return None
    target = Path(directory) / f"replaced-{label}"
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        os.replace(path, target / path.name)
    return target


def load_latest_checkpoint(trainer, directory: str | Path) -> Path | None:
    """Resume ``trainer`` from the newest readable checkpoint in ``directory``.

    A file that cannot be read at all (cut short by a full disk, say) is
    skipped with a warning and the one before it is tried. A file that reads
    but does not fit the config still stops the run, as in load_checkpoint.
    """
    for path in reversed(list_checkpoints(directory)):
        try:
            state = torch.load(path, map_location=trainer.device, weights_only=True)
        except Exception as error:  # noqa: BLE001 - any unreadable file is skipped, loudly
            reason = (str(error).splitlines() or [""])[0][:160]
            print(f"warning: {path} cannot be read ({type(error).__name__}: {reason}); trying the one before it")
            continue
        _restore(trainer, state, path)
        return path
    return None


def load_checkpoint(trainer, path: str | Path) -> None:
    """Restore models, optimizer and schedule state into ``trainer``."""
    state = torch.load(path, map_location=trainer.device, weights_only=True)
    _restore(trainer, state, path)


def _restore(trainer, state: dict, path: str | Path) -> None:
    band = trainer.models.band.as_dict()
    saved_band = state["band"]
    for key in ("grating_step", "message_grid", "texture_kernel", "texture_downsample"):
        if saved_band[key] != band[key]:
            raise RuntimeError(
                f"{path} was trained with {key}={saved_band[key]} but the config now gives {band[key]}. "
                f"If camera_px_per_tile changed since the run started, set band.grating_step: "
                f"{saved_band['grating_step']} to continue this run, or start a new one in another directory."
            )
    names = set(trainer.models.named())
    if set(state["models"]) != names:
        raise RuntimeError(f"{path} holds models {sorted(state['models'])}, the config builds {sorted(names)}")
    for name, module in trainer.models.named().items():
        module.load_state_dict(state["models"][name])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.step = int(state["step"])
    trainer.acc_ema = float(state["acc_ema"])
    trainer.image_ramp_start = state["image_ramp_start"]
    # A checkpoint from before the weight was saved continues from where the plain ramp stood.
    trainer.follow_scale = float(state["image_scale"]) if "image_scale" in state else trainer.ramp_scale()
    # A checkpoint from before follow_visibility starts the controller over.
    follow_visibility = trainer.visibility_follow_settings()
    start = follow_visibility["init_weight"] if follow_visibility else 0.01
    trainer.vis_weight = float(state["vis_weight"]) if state.get("vis_weight") is not None else start
    trainer.vis_ema = float(state["vis_ema"]) if state.get("vis_ema") is not None else None
