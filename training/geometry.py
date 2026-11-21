"""Homographies and the flatten step.

Coordinates are continuous pixel coordinates (x, y): the centre of the pixel
in column j, row i is (j + 0.5, i + 0.5), and an image W wide spans x in
[0, W]. A homography is a (B,3,3) tensor acting on column vectors (x, y, 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def homography_from_points(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Homography that maps four points ``src`` (B,4,2) onto ``dst`` (B,4,2).

    Solves the usual 8x8 linear system with the bottom-right entry fixed to 1.
    """
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    rows_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    a = torch.cat([rows_u, rows_v], dim=1)                       # (B,8,8)
    b = torch.cat([u, v], dim=1).unsqueeze(-1)                   # (B,8,1)
    # Solve in float64 on the CPU: tiny systems, and MPS has no float64.
    h = torch.linalg.solve(a.double().cpu(), b.double().cpu()).squeeze(-1)
    h = torch.cat([h, torch.ones_like(h[:, :1])], dim=1).view(-1, 3, 3)
    return h.to(device=src.device, dtype=src.dtype)


def apply_homography(h: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Map points (B,N,2) through homographies (B,3,3)."""
    ones = torch.ones_like(points[..., :1])
    mapped = torch.cat([points, ones], dim=-1) @ h.transpose(1, 2)
    return mapped[..., :2] / mapped[..., 2:3]


def warp(image: torch.Tensor, out_to_in: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Resample ``image`` (B,C,H,W) into an out_hw image.

    ``out_to_in`` maps output pixel coordinates to input pixel coordinates.
    Bilinear, differentiable with respect to the image; outside the input the
    result is 0.
    """
    b, _, h, w = image.shape
    oh, ow = out_hw
    ys = torch.arange(oh, device=image.device, dtype=image.dtype) + 0.5
    xs = torch.arange(ow, device=image.device, dtype=image.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx, yy], dim=-1).view(1, -1, 2).expand(b, -1, -1)
    src = apply_homography(out_to_in, points)
    scale = torch.tensor([2.0 / w, 2.0 / h], device=image.device, dtype=image.dtype)
    grid = (src * scale - 1.0).view(b, oh, ow, 2)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def warp_periodic(image: torch.Tensor, out_to_in: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Like ``warp``, for an image that repeats without end in both directions.

    ``image`` (B,C,H,W) is one period. Input coordinates are wrapped into it,
    so nothing is ever outside, and the blend across the seam uses the pixels
    from the other edge. Used for the tiled surface, which a view at a small
    scale or a steep tilt sees far more of than any fixed number of repeats.
    """
    b, _, h, w = image.shape
    oh, ow = out_hw
    ys = torch.arange(oh, device=image.device, dtype=image.dtype) + 0.5
    xs = torch.arange(ow, device=image.device, dtype=image.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx, yy], dim=-1).view(1, -1, 2).expand(b, -1, -1)
    src = apply_homography(out_to_in, points)
    # One wrapped pixel on every side, so bilinear sampling near an edge finds its neighbour.
    padded = F.pad(image, (1, 1, 1, 1), mode="circular")
    x = torch.remainder(src[..., 0], w) + 1.0
    y = torch.remainder(src[..., 1], h) + 1.0
    grid = torch.stack([x * (2.0 / (w + 2)) - 1.0, y * (2.0 / (h + 2)) - 1.0], dim=-1).view(b, oh, ow, 2)
    return F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=False)


def square_corners(size: float, batch: int, device=None, dtype=None) -> torch.Tensor:
    """Corners of a size x size square: top-left, top-right, bottom-right, bottom-left. (B,4,2)."""
    c = torch.tensor([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], device=device, dtype=dtype)
    return c.unsqueeze(0).repeat(batch, 1, 1)


def flatten_from_corners(frame: torch.Tensor, corners: torch.Tensor, out_px: int) -> torch.Tensor:
    """Flatten the quadrilateral ``corners`` (B,4,2, frame pixels) to out_px x out_px.

    Corner order: top-left, top-right, bottom-right, bottom-left. This is the
    function the locator's output will go through at run time.
    """
    patch = square_corners(float(out_px), frame.shape[0], frame.device, frame.dtype)
    return warp(frame, homography_from_points(patch, corners), (out_px, out_px))


def flatten_random_patch(
    frame: torch.Tensor,
    surface_to_frame: torch.Tensor,
    tile_px: int,
    corner_jitter_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten one tile-sized patch of the surface out of ``frame``.

    The patch is centred on the surface point seen at the middle of the
    frame, plus an offset drawn uniformly over one full tile period in both
    directions, so the reader sees every offset within the tile pattern. Its
    corners are carried into the frame by the true homography
    ``surface_to_frame`` (surface pixels -> frame pixels), each corner is
    then moved by up to ``corner_jitter_px`` frame pixels in x and y, and the
    resulting quadrilateral is flattened to tile_px x tile_px.

    The patch reaches at most one tile from the surface point at the middle
    of the frame, so the frame has to show two tiles' worth of surface around
    its middle. The simulation guarantees that.

    Returns (patch, corners in frame pixels after jitter).
    """
    b, _, h, w = frame.shape
    middle = torch.tensor([[[w / 2.0, h / 2.0]]], device=frame.device, dtype=frame.dtype).expand(b, -1, -1)
    frame_to_surface = torch.linalg.inv(surface_to_frame.double().cpu()).to(surface_to_frame)
    seen_at_middle = apply_homography(frame_to_surface, middle)                      # (B,1,2)
    top_left = seen_at_middle - torch.rand(b, 1, 2, device=frame.device, dtype=frame.dtype) * tile_px
    corners_surface = top_left + square_corners(float(tile_px), b, frame.device, frame.dtype)
    corners = apply_homography(surface_to_frame, corners_surface)
    if corner_jitter_px > 0:
        corners = corners + (torch.rand_like(corners) * 2.0 - 1.0) * corner_jitter_px
    return flatten_from_corners(frame, corners, tile_px), corners


def center_crop(frame: torch.Tensor, size: int) -> torch.Tensor:
    h, w = frame.shape[-2:]
    if h < size or w < size:
        raise ValueError(f"cannot take a {size}px centre crop from {h}x{w}")
    top, left = (h - size) // 2, (w - size) // 2
    return frame[..., top : top + size, left : left + size]
