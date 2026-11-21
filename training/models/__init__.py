"""The three models: embedder, reader and (slot only, for now) locator."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

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

    def named(self) -> dict[str, nn.Module]:
        """The models that exist, by name. Used for checkpoints and device moves."""
        out: dict[str, nn.Module] = {"embedder": self.embedder, "reader": self.reader}
        if self.locator is not None:
            out["locator"] = self.locator
        return out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_models(cfg: dict) -> Models:
    m = cfg["model"]
    emb, rd = m["embedder"], m["reader"]
    embedder = Embedder(
        message_bits=MESSAGE_BITS,
        tile_px=TILE_PX,
        base_channels=emb["base_channels"],
        message_grid=emb["message_grid"],
        message_channels=emb["message_channels"],
        message_init=emb["message_init"],
        grating_step=emb["grating_step"],
        alpha=emb["alpha"],
    )
    texture = rd.get("texture")
    reader = Reader(
        message_bits=MESSAGE_BITS,
        widths=tuple(rd["widths"]),
        texture=dict(texture) if texture else None,
    )
    locator = Locator(**m["locator"]) if m.get("locator") else None
    return Models(embedder=embedder, reader=reader, locator=locator)
