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


def add_footer(image: torch.Tensor, text: str | None) -> torch.Tensor:
    """The image with a white strip below it carrying ``text``; unchanged if there is no text.

    For the credit line a sheet of dataset photos has to show (data.credit).
    The strip is as wide as the image; the type is sized to the width and
    shrunk until the line fits.
    """
    if not text:
        return image
    if not isinstance(text, str):
        raise ValueError(f"the footer (data.credit) must be a line of text, got {text!r}; quote it in the config")
    from PIL import ImageDraw, ImageFont

    _, _, width = image.shape
    size = max(12, min(40, width // 60))
    while True:
        font = ImageFont.load_default(size=size)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= width - 2 * size or size <= 8:
            break
        size -= 1
    height = (bottom - top) + size
    strip = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(strip).text((size // 2, size // 2 - top), text, fill=(0, 0, 0), font=font)
    data = torch.frombuffer(bytearray(strip.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    footer = data.permute(2, 0, 1).float() / 255.0
    return torch.cat([image.detach().float().cpu().clamp(0.0, 1.0), footer], dim=1)


def save_image(image: torch.Tensor, path: str | Path, footer: str | None = None) -> Path:
    """Save a (3,H,W) tensor in [0,1] as PNG, with ``footer`` as a line of text below it if given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = add_footer(image, footer)
    array = (image.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)
    return path
