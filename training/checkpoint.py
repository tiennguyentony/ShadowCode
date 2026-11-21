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


def load_checkpoint(trainer, path: str | Path) -> None:
    """Restore models, optimizer and schedule state into ``trainer``."""
    state = torch.load(path, map_location=trainer.device, weights_only=True)
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
