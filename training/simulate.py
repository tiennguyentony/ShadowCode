"""Surface simulation between encoder and decoder.

Only the parts the smoke test needs live here so far: tiling the encoded
image into a wrapped surface and taking a random crop of it.
"""

from __future__ import annotations

import torch


def tile(x: torch.Tensor, n: int = 3) -> torch.Tensor:
    """Repeat (B,C,H,W) n times in both directions: (B,C,n*H,n*W)."""
    return x.repeat(1, 1, n, n)


def random_crop(x: torch.Tensor, size: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Independent random size x size crop for every item in the batch."""
    b, _, h, w = x.shape
    if h < size or w < size:
        raise ValueError(f"cannot crop {size} from {h}x{w}")
    tops = torch.randint(0, h - size + 1, (b,), generator=generator).tolist()
    lefts = torch.randint(0, w - size + 1, (b,), generator=generator).tolist()
    return torch.stack([x[i, :, t : t + size, l : l + size] for i, (t, l) in enumerate(zip(tops, lefts))])


def simulate(encoded: torch.Tensor, cfg: dict, strength: float) -> torch.Tensor:
    """Encoded image batch -> what the decoder sees.

    With distortions disabled this is a 3x3 tiling followed by a random
    crop at an arbitrary tile offset, with no resampling.
    """
    dist = cfg["distortion"]
    crop = cfg["model"]["decoder"]["input_size"]
    surface = tile(encoded, int(dist.get("tile", 3)))
    if not dist.get("enabled", True):
        return random_crop(surface, crop)
    raise NotImplementedError("print-and-camera distortions are not written yet")
