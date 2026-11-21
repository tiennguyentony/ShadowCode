"""The three models: embedder, reader and (slot only, for now) locator."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from band import Band, resolve_band
from constants import MESSAGE_BITS, TILE_PX

from .embedder import Embedder
from .locator import Locator
from .reader import Reader

__all__ = ["Embedder", "Reader", "Locator", "Models", "build_models", "count_parameters"]


@dataclass
class Models:
    embedder: Embedder
    reader: Reader
    locator: Locator | None = None
    band: Band | None = None

    def named(self) -> dict[str, nn.Module]:
        """The models that exist, by name. Used for checkpoints and device moves."""
        out: dict[str, nn.Module] = {"embedder": self.embedder, "reader": self.reader}
        if self.locator is not None:
            out["locator"] = self.locator
        return out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_models(cfg: dict) -> Models:
    """Build the models. The band (see band.py) fixes the sizes that depend on it."""
    m = cfg["model"]
    emb, rd = m["embedder"], m["reader"]
    band = resolve_band(cfg)
    embedder = Embedder(
        message_bits=MESSAGE_BITS,
        tile_px=TILE_PX,
        base_channels=emb["base_channels"],
        message_grid=band.message_grid,
        message_channels=emb["message_channels"],
        message_init=emb["message_init"],
        grating_step=band.grating_step,
        alpha=emb["alpha"],
    )
    texture = None
    if rd.get("texture"):
        texture = {
            "filters": rd["texture"]["filters"],
            "stride": rd["texture"]["stride"],
            "kernel": band.texture_kernel,
            "downsample": band.texture_downsample,
        }
    reader = Reader(
        message_bits=MESSAGE_BITS,
        widths=tuple(rd["widths"]),
        texture=texture,
        pool_norm=rd["pool_norm"],
    )
    locator = Locator(**m["locator"]) if m.get("locator") else None
    return Models(embedder=embedder, reader=reader, locator=locator, band=band)
