"""Image losses between the marked tile and the original."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Used when the config has no loss.texture_mask (configs and checkpoints from before it existed).
TEXTURE_DEFAULTS = {"enabled": False, "flat_weight": 1.5, "busy_weight": 0.25, "level": 6.0e-5, "window_px": 17}


def texture_settings(cfg: dict) -> dict:
    """``loss.texture_mask`` merged over the defaults, checked."""
    s = {**TEXTURE_DEFAULTS, **(cfg.get("loss", {}).get("texture_mask") or {})}
    if s["enabled"]:
        if not 0.0 < float(s["busy_weight"]) <= float(s["flat_weight"]):
            raise ValueError(f"loss.texture_mask needs 0 < busy_weight <= flat_weight, got {s}")
        if float(s["level"]) <= 0.0:
            raise ValueError(f"loss.texture_mask.level must be above 0, got {s['level']}")
        if int(s["window_px"]) < 1 or int(s["window_px"]) % 2 == 0:
            raise ValueError(f"loss.texture_mask.window_px must be odd, got {s['window_px']}")
    return s


@torch.no_grad()
def texture_weights(tiles: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Per-pixel cost of the mark, (B,1,H,W), from the clean tile: high on flat colour, low on busy detail.

    Local contrast is measured as in the spread check and acceptance.py: the
    grey image minus a 9 px blur of itself, squared, here averaged over a
    ``window_px`` square around each pixel (padded by reflection, so the tile
    edge does not count as detail). The weight is
    ``busy + (flat - busy) * level / (level + contrast)``: ``flat_weight`` where
    the tile is perfectly flat, half-way at contrast ``level``, falling to
    ``busy_weight`` on text, edges and texture. ``level`` 6e-5 is acceptance.py's
    flat level (about 2 grey levels of 255).
    """
    s = texture_settings(cfg)
    grey = tiles.mean(dim=1, keepdim=True)
    blurred = F.avg_pool2d(F.pad(grey, (4, 4, 4, 4), mode="reflect"), 9, 1)
    detail = (grey - blurred).pow(2)
    half = int(s["window_px"]) // 2
    contrast = F.avg_pool2d(F.pad(detail, (half, half, half, half), mode="reflect"), int(s["window_px"]), 1)
    flat, busy, level = float(s["flat_weight"]), float(s["busy_weight"]), float(s["level"])
    return busy + (flat - busy) * level / (level + contrast)


class PerceptualLoss(nn.Module):
    """LPIPS distance (Zhang et al., 2018), frozen. Inputs are (B,3,H,W) in [0,1].

    Uses the ``lpips`` package. Its backbone weights are pretrained and are
    downloaded by torchvision on first use. See THIRD_PARTY.md.
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        try:
            import lpips
        except ImportError as error:  # pragma: no cover
            raise ImportError("loss.lpips_weight > 0 needs the lpips package: pip install lpips") from error
        self.model = lpips.LPIPS(net=net, verbose=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "PerceptualLoss":
        return super().train(False)  # always in eval mode

    def forward(self, marked: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.model(marked * 2.0 - 1.0, original * 2.0 - 1.0).mean()
