"""Keeping the mark spread over the whole tile.

Left alone, the image loss lets the embedder put the whole code into one
bright patch: the mean-square loss does not care where the energy sits, and
the reader always sees a full tile, so one spot is enough. Two switches in the
config section ``spread`` work against that, one on each side:

* ``peak_penalty`` (embedder side): a loss on the strongest local region of
  the residual. A patch costs far more than the same energy spread evenly.
* ``hide`` (reader side): random regions of the reader's patch are blanked
  out during training, so no single spot can be relied on.

Both are off by default. ``spread_report`` is the measurement to watch either
way: above all its ``fixed_place``, which is high only for a patch at the same
place in every tile and stays low for a mark that follows the image content.
"""

from __future__ import annotations

import math

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


def fixed_batch(image_dir, count: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``count`` dataset images and messages that are the same in every run and at every step.

    Images are picked by a seeded shuffle of the sorted file list, resized on
    the short side and centre-cropped to the tile. Returns (tiles, bits) on the CPU.
    """
    from torchvision import transforms

    from constants import MESSAGE_BITS, TILE_PX
    from data import list_images, load_rgb

    files = list_images(image_dir)
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(files), generator=generator)[:count].tolist()
    cut = transforms.Compose(
        [transforms.Resize(TILE_PX, antialias=True), transforms.CenterCrop(TILE_PX), transforms.ToTensor()]
    )
    tiles = []
    for index in picks:
        tiles.append(cut(load_rgb(files[index])))
    bits = torch.randint(0, 2, (len(picks), MESSAGE_BITS), generator=generator).float()
    return torch.stack(tiles), bits


def busy_windows(tiles: torch.Tensor, cfg: dict) -> torch.Tensor:
    """For each image, which windows are its busier half: (B, rows, cols) of bool.

    Busy means local contrast (the image minus a 9 px blur of itself, squared,
    averaged over the window) above that image's median. The blur is padded by
    reflection, so the tile edge does not count as contrast.
    """
    s = peak_settings(cfg)
    grey = tiles.mean(dim=1, keepdim=True)
    blurred = F.avg_pool2d(F.pad(grey, (4, 4, 4, 4), mode="reflect"), 9, 1)
    contrast = local_energy(grey - blurred, int(s["window_px"]), int(s["stride_px"]))
    return contrast > contrast.flatten(1).median(dim=1).values[:, None, None]


@torch.no_grad()
def spread_report(
    embedder, tiles: torch.Tensor, bits: torch.Tensor, cfg: dict, busy: torch.Tensor | None = None, chunk: int = 8
) -> dict[str, float]:
    """Where the embedder's mark sits on the fixed images, in eval mode.

    * ``fixed_place``: is the mark at the same place in every tile? Each
      image's local-energy map is scaled to mean 1, the maps are averaged
      over the images, and this is the mean of the strongest windows of that
      average. A patch at one place in every tile keeps its full height (the
      long run's corner patch: 18 to 20). A mark that only follows each
      image's content mostly averages out (1.4 to 2.7 for made-up marks of
      that kind on 256 images; docs/runs.md has the baseline). It needs many
      images: on 16 the same marks reach 6.
    * ``strongest_spot``: as ``concentration``, per image, averaged. 1 is
      even. It cannot tell a fixed patch from a mark that follows the image.
    * ``square_share``: share of the mark's energy inside its strongest
      SQUARE_PX square, averaged over the images. An even mark gives the
      square's share of the tile area (6.25 percent).
    * ``busy_share``: share of the mark's energy on the busier half of each
      image (``busy`` from ``busy_windows``). 0.5 means the mark ignores the image.
    * ``psnr``: marked against original, over all the images.

    ``tiles`` and ``bits`` may stay on the CPU; they are moved to the
    embedder's device a chunk at a time. The embedder is left in training mode.
    """
    s = peak_settings(cfg)
    window, stride, top = int(s["window_px"]), int(s["stride_px"]), float(s["top_fraction"])
    device = next(embedder.parameters()).device
    square = min(SQUARE_PX, tiles.shape[-1])
    area = square * square / (tiles.shape[-1] * tiles.shape[-2])
    count = len(tiles)
    totals = {"strongest_spot": 0.0, "square_share": 0.0, "busy_share": 0.0, "squared_error": 0.0}
    map_sum = None
    embedder.eval()
    for i in range(0, count, chunk):
        tile = tiles[i : i + chunk].to(device)
        residual = embedder(tile, bits[i : i + chunk].to(device))
        mean = residual.pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        energy = local_energy(residual, window, stride)
        flat = energy.flatten(1)
        strongest = flat.topk(max(1, round(top * flat.shape[1])), dim=1).values.mean(dim=1)
        totals["strongest_spot"] += (strongest / mean).sum().item()
        in_square = local_energy(residual, square, stride).flatten(1).max(dim=1).values
        totals["square_share"] += (in_square * area / mean).sum().item()
        if busy is not None:
            on_busy = (energy * busy[i : i + chunk].to(device)).sum(dim=(1, 2)) / energy.sum(dim=(1, 2)).clamp_min(1e-12)
            totals["busy_share"] += on_busy.sum().item()
        totals["squared_error"] += ((tile + residual).clamp(0.0, 1.0) - tile).pow(2).mean(dim=(1, 2, 3)).sum().item()
        scaled = (energy / mean[:, None, None]).sum(dim=0)
        map_sum = scaled if map_sum is None else map_sum + scaled
    embedder.train()
    mean_map = (map_sum / count).flatten()
    fixed = mean_map.topk(max(1, round(top * mean_map.numel()))).values.mean() / mean_map.mean()
    return {
        "images": count,
        "fixed_place": fixed.item(),
        "strongest_spot": totals["strongest_spot"] / count,
        "square_share": totals["square_share"] / count,
        "busy_share": totals["busy_share"] / count if busy is not None else float("nan"),
        "psnr": -10.0 * math.log10(max(totals["squared_error"] / count, 1e-10)),
    }
