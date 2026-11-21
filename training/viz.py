"""Sample images: original, marked, residual x10."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

RESIDUAL_GAIN = 10.0


def residual_image(residual: torch.Tensor, gain: float = RESIDUAL_GAIN) -> torch.Tensor:
    """Residual shown around mid-grey and amplified: 0.5 + gain * residual, clipped to [0,1]."""
    return (0.5 + gain * residual).clamp(0.0, 1.0)


def sample_sheet(original: torch.Tensor, marked: torch.Tensor, gap: int = 8) -> torch.Tensor:
    """One row per sample: original | marked | residual x10. Returns (3,H,W) in [0,1]."""
    original, marked = original.detach().float().cpu(), marked.detach().float().cpu()
    residual = residual_image(marked - original)
    b, c, h, w = original.shape
    spacer = torch.ones(b, c, h, gap)
    rows = torch.cat([original, spacer, marked, spacer, residual], dim=3)
    line = torch.ones(c, gap, rows.shape[-1])
    parts = []
    for i in range(b):
        parts.extend([rows[i], line])
    return torch.cat(parts[:-1], dim=1).clamp(0.0, 1.0)


def save_image(image: torch.Tensor, path: str | Path) -> Path:
    """Save a (3,H,W) tensor in [0,1] as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)
    return path
