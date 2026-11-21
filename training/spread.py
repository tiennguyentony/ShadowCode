"""Keeping the mark spread over the whole tile.

Left alone, the image loss lets the embedder put the whole code into one
bright patch: the mean-square loss does not care where the energy sits, and
the reader always sees a full tile, so one spot is enough. Two switches in the
config section ``spread`` work against that, one on each side:

* ``peak_penalty`` (embedder side): a loss on the strongest local region of
  the residual. A patch costs far more than the same energy spread evenly.
* ``hide`` (reader side): random regions of the reader's patch are blanked
  out during training, so no single spot can be relied on.

Both are off by default. ``concentration`` is the number to watch either way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Used when the config has no ``spread`` section (checkpoints saved before it existed).
PEAK_DEFAULTS = {"enabled": False, "weight": 4.0, "window_px": 48, "stride_px": 8, "top_fraction": 0.02}
HIDE_DEFAULTS = {"enabled": False, "max_fraction": 0.5, "cells": [2, 3, 4, 6, 8]}


def peak_settings(cfg: dict) -> dict:
    return {**PEAK_DEFAULTS, **((cfg.get("spread") or {}).get("peak_penalty") or {})}


def hide_settings(cfg: dict) -> dict:
    return {**HIDE_DEFAULTS, **((cfg.get("spread") or {}).get("hide") or {})}


def local_energy(residual: torch.Tensor, window_px: int, stride_px: int) -> torch.Tensor:
    """Mean square of the residual in every window, as (B, rows, cols).

    The tile is a torus: windows run over the edges and come back on the
    other side, so a patch sitting on a corner is one patch, not four quarters.
    """
    height, width = residual.shape[-2:]
    if not 0 < window_px <= min(height, width):
        raise ValueError(f"spread.peak_penalty.window_px must be between 1 and {min(height, width)}")
    if stride_px <= 0 or height % stride_px or width % stride_px:
        raise ValueError(f"spread.peak_penalty.stride_px must divide the tile size {height}")
    energy = residual.pow(2).mean(dim=1, keepdim=True)
    # Windows start every stride_px pixels; the last ones need window - stride pixels from the far side.
    reach = max(0, window_px - stride_px)
    reach_x, reach_y = min(reach, width), min(reach, height)
    if reach:
        energy = F.pad(energy, (0, reach_x, 0, reach_y), mode="circular")
    return F.avg_pool2d(energy, window_px, stride_px).squeeze(1)


def strongest_windows(residual: torch.Tensor, window_px: int, stride_px: int, top_fraction: float) -> torch.Tensor:
    """Mean energy of the strongest ``top_fraction`` of windows, one value per sample."""
    energy = local_energy(residual, window_px, stride_px).flatten(1)
    count = max(1, round(float(top_fraction) * energy.shape[1]))
    return energy.topk(count, dim=1).values.mean(dim=1)


def peak_penalty(residual: torch.Tensor, cfg: dict) -> torch.Tensor:
    """The loss: energy of the strongest windows, averaged over the batch.

    For an evenly spread mark this equals the plain mean-square loss. For a
    mark packed into a twentieth of the tile it is about twenty times that.
    """
    s = peak_settings(cfg)
    return strongest_windows(residual, int(s["window_px"]), int(s["stride_px"]), float(s["top_fraction"])).mean()


@torch.no_grad()
def concentration(residual: torch.Tensor, cfg: dict) -> float:
    """How piled-up the mark is: strongest windows over the tile average.

    1 is perfectly even. Measured with the windows of ``peak_penalty`` whether
    or not the penalty is on.
    """
    s = peak_settings(cfg)
    peak = strongest_windows(residual, int(s["window_px"]), int(s["stride_px"]), float(s["top_fraction"]))
    mean = residual.pow(2).mean(dim=(1, 2, 3))
    return (peak / mean.clamp_min(1e-12)).mean().item()


def hide_mask(batch: int, height: int, width: int, max_fraction: float, cells: list[int]) -> torch.Tensor:
    """(B,1,H,W) mask on the CPU: 1 where the reader may look, 0 where it is blanked.

    Per sample: the patch is cut into n x n cells (n drawn from ``cells``),
    a share of them drawn uniformly from 0 to ``max_fraction`` is blanked, and
    the grid is shifted by a random amount so the cell edges fall anywhere.
    Few cells give large connected regions, many cells give scattered ones.
    """
    if not 0.0 <= max_fraction < 1.0:
        raise ValueError("spread.hide.max_fraction must be at least 0 and below 1")
    keep = torch.ones(batch, 1, height, width)
    for i in range(batch):
        n = int(cells[torch.randint(len(cells), (1,)).item()])
        # Rounded to whole cells, and never past the limit.
        count = min(round(torch.rand(1).item() * max_fraction * n * n), int(max_fraction * n * n + 1e-9))
        if count == 0:
            continue
        grid = torch.ones(n * n)
        grid[torch.randperm(n * n)[:count]] = 0.0
        mask = F.interpolate(grid.view(1, 1, n, n), size=(height, width), mode="nearest")
        shift = (torch.randint(height, (1,)).item(), torch.randint(width, (1,)).item())
        keep[i] = torch.roll(mask, shift, dims=(2, 3))[0]
    return keep


def hide_regions(view: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, float]:
    """Blank random regions of the reader's patch. Returns the patch and the share blanked.

    Blanked pixels take the patch's own mean colour, so the reader gets no
    mark there and no bright edge either. Pixels that stay are untouched, and
    the gradient reaches the embedder through them only.
    """
    s = hide_settings(cfg)
    b, _, h, w = view.shape
    keep = hide_mask(b, h, w, float(s["max_fraction"]), list(s["cells"])).to(view)
    fill = view.detach().mean(dim=(2, 3), keepdim=True)
    return view * keep + fill * (1.0 - keep), 1.0 - keep.mean().item()


# -- the spread check: the same measurement at every sample sheet --------------

SQUARE_PX = 96  # the "one square" of the check: 1/16 of the tile area at TILE_PX 384


def fixed_batch(image_dir, count: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``count`` dataset images and messages that are the same in every run and at every step.

    Images are picked by a seeded shuffle of the sorted file list, resized on
    the short side and centre-cropped to the tile. Returns (tiles, bits) on the CPU.
    """
    from PIL import Image
    from torchvision import transforms

    from constants import MESSAGE_BITS, TILE_PX
    from data import list_images

    files = list_images(image_dir)
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(files), generator=generator)[:count].tolist()
    cut = transforms.Compose(
        [transforms.Resize(TILE_PX, antialias=True), transforms.CenterCrop(TILE_PX), transforms.ToTensor()]
    )
    tiles = []
    for index in picks:
        with Image.open(files[index]) as image:
            tiles.append(cut(image.convert("RGB")))
    bits = torch.randint(0, 2, (len(picks), MESSAGE_BITS), generator=generator).float()
    return torch.stack(tiles), bits


@torch.no_grad()
def spread_report(embedder, tiles: torch.Tensor, bits: torch.Tensor, cfg: dict, chunk: int = 4) -> dict[str, float]:
    """How piled-up the embedder's mark is on the fixed images, in eval mode.

    * ``strongest_spot``: as ``concentration``; 1 is even.
    * ``square_share``: share of the mark's energy inside its strongest
      SQUARE_PX square, averaged over the images. An even mark gives the
      square's share of the tile area (6.25 percent).
    * ``psnr``: marked against original, over all the images.

    The embedder is left in training mode.
    """
    embedder.eval()
    residual = torch.cat([embedder(tiles[i : i + chunk], bits[i : i + chunk]) for i in range(0, len(tiles), chunk)])
    embedder.train()
    mean = residual.pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    s = peak_settings(cfg)
    square = min(SQUARE_PX, residual.shape[-1])
    in_square = local_energy(residual, square, int(s["stride_px"])).flatten(1).max(dim=1).values
    area = square * square / (residual.shape[-1] * residual.shape[-2])
    mse = ((tiles + residual).clamp(0.0, 1.0) - tiles).pow(2).mean().clamp_min(1e-10)
    return {
        "strongest_spot": concentration(residual, cfg),
        "square_share": (in_square * area / mean).mean().item(),
        "psnr": (-10.0 * torch.log10(mse)).item(),
    }
