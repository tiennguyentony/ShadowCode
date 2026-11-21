"""Differentiable print-and-camera simulation between the embedder and the reader.

``simulate`` takes the wrapped surface (the marked tile repeated) and returns
the camera frame together with the exact homography it applied, mapping
surface pixels to frame pixels (coordinates as in geometry.py).

One knob, ``strength`` in [0, 1], scales every range. Order of the steps:

1. perspective warp           tilt up to max_tilt_deg * strength
   (direct mode only: plus random rotation, scale and shift)
2. defocus and motion blur
3. colour shift and contrast
4. loss of resolution, down   to camera_px_per_tile pixels per tile
5. sensor noise               at camera resolution
6. approximate JPEG           at camera resolution
7. loss of resolution, up     back to the frame size

Steps 4 and 7 are skipped while ``camera_px_per_tile`` is unset; everything
else still runs. Each step can be switched off in ``simulation.effects``.
Everything is differentiable with respect to the surface.

The frame is a window onto the surface drawn at TILE_PX pixels per tile. It
is not the camera stream: the resolution of the camera is what steps 4 to 7
model. With ``camera_px_per_tile: [low, high]`` every sample draws its own
value, log-uniform over the range. The frame is square by default, which is
neutral between portrait and landscape; ``simulation.frame_px: [width,
height]`` gives any other shape, and nothing here or in geometry.py depends
on which side is longer.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import torch
import torch.nn.functional as F

from camera import camera_range, frame_size
from constants import TILE_PX
from geometry import apply_homography, warp


class SimResult(NamedTuple):
    frame: torch.Tensor        # (B,3,H,W) in [0,1]: what the camera sees
    homography: torch.Tensor   # (B,3,3): surface pixels -> frame pixels, exact


def wrap_surface(marked: torch.Tensor, repeat: int = 3) -> torch.Tensor:
    """Repeat the marked tile repeat x repeat times: (B,C,H,W) -> (B,C,repeat*H,repeat*W)."""
    return marked.repeat(1, 1, repeat, repeat)


# -- geometry ---------------------------------------------------------------


def _translate(tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    h = torch.eye(3, dtype=torch.float64).repeat(tx.shape[0], 1, 1)
    h[:, 0, 2], h[:, 1, 2] = tx, ty
    return h


def _scale_rotate(scale: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    h = torch.eye(3, dtype=torch.float64).repeat(scale.shape[0], 1, 1)
    cos, sin = scale * torch.cos(angle), scale * torch.sin(angle)
    h[:, 0, 0], h[:, 0, 1], h[:, 1, 0], h[:, 1, 1] = cos, -sin, sin, cos
    return h


def _tilt(tilt: torch.Tensor, azimuth: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Pinhole view of a plane tilted by ``tilt`` about an in-plane axis at ``azimuth``.

    The camera is ``distance`` pixels from the plane's centre with a focal
    length equal to that distance, so the centre keeps scale 1. Acts on
    centred surface coordinates.
    """
    ax, ay = torch.cos(azimuth), torch.sin(azimuth)
    c, s = torch.cos(tilt), torch.sin(tilt)
    # Rodrigues rotation about (ax, ay, 0); only the first two columns matter for z = 0.
    r11, r12 = c + ax * ax * (1 - c), ax * ay * (1 - c)
    r21, r22 = ax * ay * (1 - c), c + ay * ay * (1 - c)
    r31, r32 = -ay * s, ax * s
    h = torch.zeros(tilt.shape[0], 3, 3, dtype=torch.float64)
    h[:, 0, 0], h[:, 0, 1] = distance * r11, distance * r12
    h[:, 1, 0], h[:, 1, 1] = distance * r21, distance * r22
    h[:, 2, 0], h[:, 2, 1], h[:, 2, 2] = r31, r32, distance
    return h


def sample_homography(
    batch: int, surface_px: int, cfg: dict, strength: float, direct: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random surface -> frame homography, (B,3,3) float64 on the CPU, and the fit scale (B,).

    The fit scale is how much each sample was shrunk to fit the frame (1 in
    direct mode). It is not part of the camera model, so the loss-of-resolution
    step divides it out again.
    """
    sim = cfg["simulation"]
    frame_w, frame_h = frame_size(cfg)
    half = surface_px / 2.0
    centre = torch.full((batch,), -half, dtype=torch.float64)
    h = _translate(centre, centre)

    if direct:
        # Random position within the tile pattern, so the centre crop starts anywhere in a tile.
        shift = (torch.rand(batch, 2, dtype=torch.float64) - 0.5) * TILE_PX
        h = _translate(shift[:, 0], shift[:, 1]) @ h

    if sim["effects"]["perspective"] and strength > 0:
        p = sim["perspective"]
        tilt = torch.rand(batch, dtype=torch.float64) * math.radians(p["max_tilt_deg"]) * strength
        azimuth = torch.rand(batch, dtype=torch.float64) * 2 * math.pi
        lo, hi = p["distance_tiles"]
        distance = (lo + torch.rand(batch, dtype=torch.float64) * (hi - lo)) * TILE_PX
        h = _tilt(tilt, azimuth, distance) @ h

    if direct:
        d = sim["direct"]
        amount = strength if d["ramp_with_strength"] else 1.0
        angle = torch.rand(batch, dtype=torch.float64) * math.radians(d["rotation_deg"]) * amount
        lo, hi = d["scale"]
        log_scale = math.log(lo) + torch.rand(batch, dtype=torch.float64) * (math.log(hi) - math.log(lo))
        h = _scale_rotate(torch.exp(log_scale * amount), angle) @ h
        fit = torch.ones(batch, dtype=torch.float64)
    else:
        # Keep the middle two tiles' worth of surface inside the frame, so a
        # tile-sized patch at any offset can be flattened out of it.
        reach = TILE_PX + 8.0
        corners = torch.tensor([[-reach, -reach], [reach, -reach], [reach, reach], [-reach, reach]], dtype=torch.float64)
        projected = apply_homography(h, (corners + half).unsqueeze(0).expand(batch, -1, -1))
        extent_x, extent_y = projected[..., 0].abs().amax(dim=1), projected[..., 1].abs().amax(dim=1)
        fit = torch.minimum((frame_w / 2.0 - 4.0) / extent_x, (frame_h / 2.0 - 4.0) / extent_y).clamp(max=1.0)
        h = _scale_rotate(fit, torch.zeros_like(fit)) @ h

    middle_x = torch.full((batch,), frame_w / 2.0, dtype=torch.float64)
    middle_y = torch.full((batch,), frame_h / 2.0, dtype=torch.float64)
    return _translate(middle_x, middle_y) @ h, fit


# -- blur -------------------------------------------------------------------


def blur_kernels(defocus_radius: torch.Tensor, motion_length: torch.Tensor, motion_angle: torch.Tensor, size: int) -> torch.Tensor:
    """One (size x size) kernel per sample: a soft disc convolved with a soft line.

    Radius 0 and length 0 give a single-pixel (identity) kernel.
    """
    b = defocus_radius.shape[0]
    r = torch.arange(size, dtype=defocus_radius.dtype, device=defocus_radius.device) - size // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    dist = torch.sqrt(xx**2 + yy**2)
    disc = (defocus_radius.view(b, 1, 1) + 0.5 - dist).clamp(0, 1)
    disc = disc / disc.sum(dim=(1, 2), keepdim=True)

    cos, sin = torch.cos(motion_angle).view(b, 1, 1), torch.sin(motion_angle).view(b, 1, 1)
    along = xx * cos + yy * sin
    across = -xx * sin + yy * cos
    line = (1 - across.abs()).clamp(0, 1) * (motion_length.view(b, 1, 1) / 2 + 0.5 - along.abs()).clamp(0, 1)
    line = line / line.sum(dim=(1, 2), keepdim=True)

    both = F.conv2d(disc.unsqueeze(0), line.unsqueeze(1), padding=size // 2, groups=b).squeeze(0)
    return both / both.sum(dim=(1, 2), keepdim=True)


def apply_blur(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Convolve each image (B,C,H,W) with its own kernel (B,k,k), reflecting at the border."""
    b, c, h, w = x.shape
    k = kernels.shape[-1]
    weight = kernels.repeat_interleave(c, dim=0).unsqueeze(1)
    padded = F.pad(x, (k // 2,) * 4, mode="reflect").reshape(1, b * c, h + k - 1, w + k - 1)
    return F.conv2d(padded, weight, groups=b * c).view(b, c, h, w)


def _blur_kernel_size(cfg_blur: dict) -> int:
    reach = cfg_blur["defocus_radius_px"] + cfg_blur["motion_length_px"] / 2.0
    return 2 * math.ceil(reach + 1) + 1


# -- approximate JPEG -------------------------------------------------------

# Example quantisation tables from the JPEG standard (ITU-T T.81, Annex K,
# tables K.1 and K.2). See THIRD_PARTY.md.
_LUMA_TABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]
_CHROMA_TABLE = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]


def _dct_matrix(device, dtype) -> torch.Tensor:
    n = torch.arange(8, device=device, dtype=dtype)
    d = torch.cos((2 * n.view(1, 8) + 1) * n.view(8, 1) * math.pi / 16) * 0.5
    d[0] = d[0] / math.sqrt(2)
    return d  # orthonormal: d @ d.T = I


def _soft_round(x: torch.Tensor) -> torch.Tensor:
    """Rounding with a usable gradient: round(x) + (x - round(x))^3."""
    nearest = torch.round(x)
    return nearest + (x - nearest) ** 3


def _quantise_plane(plane: torch.Tensor, table: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
    """8x8 block DCT, quantise, dequantise, inverse DCT. plane: (B,H,W), table: (B,8,8)."""
    b, h, w = plane.shape
    blocks = plane.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
    coeff = dct @ blocks @ dct.T
    step = table.view(b, 1, 1, 8, 8)
    coeff = _soft_round(coeff / step) * step
    blocks = dct.T @ coeff @ dct
    return blocks.permute(0, 1, 3, 2, 4).reshape(b, h, w)


def jpeg_approx(x: torch.Tensor, quality: torch.Tensor, chroma_subsampling: bool = True) -> torch.Tensor:
    """Differentiable stand-in for JPEG compression. x: (B,3,H,W) in [0,1], quality: (B,) in (0,100]."""
    b, _, h, w = x.shape
    pad_h, pad_w = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") * 255.0
    r, g, bl = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl - 128.0
    cb = -0.168736 * r - 0.331264 * g + 0.5 * bl
    cr = 0.5 * r - 0.418688 * g - 0.081312 * bl

    # Quality to table scale, as libjpeg does it.
    scale = torch.where(quality < 50, 5000.0 / quality, 200.0 - 2.0 * quality).view(b, 1, 1) / 100.0
    luma = (torch.tensor(_LUMA_TABLE, device=x.device, dtype=x.dtype) * scale).clamp(1.0, 255.0)
    chroma = (torch.tensor(_CHROMA_TABLE, device=x.device, dtype=x.dtype) * scale).clamp(1.0, 255.0)
    dct = _dct_matrix(x.device, x.dtype)

    y = _quantise_plane(y, luma, dct)
    if chroma_subsampling:
        cb = F.avg_pool2d(cb.unsqueeze(1), 2).squeeze(1)
        cr = F.avg_pool2d(cr.unsqueeze(1), 2).squeeze(1)
    cb = _quantise_plane(cb, chroma, dct)
    cr = _quantise_plane(cr, chroma, dct)
    if chroma_subsampling:
        cb = F.interpolate(cb.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
        cr = F.interpolate(cr.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)

    y = y + 128.0
    out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], dim=1) / 255.0
    return out[:, :, :h, :w]


# -- colour, noise, resolution ----------------------------------------------


def colour_shift(x: torch.Tensor, c: dict, strength: float) -> torch.Tensor:
    """Gamma, contrast about the image mean, brightness and a per-channel shift."""
    b = x.shape[0]

    def signed(*shape: int) -> torch.Tensor:
        return torch.rand(b, *shape, device=x.device, dtype=x.dtype) * 2 - 1

    gamma = torch.exp(signed(1, 1, 1) * c["gamma"] * strength)
    x = x.clamp(1e-4, 1.0) ** gamma
    contrast = 1.0 + signed(1, 1, 1) * c["contrast"] * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x + signed(1, 1, 1) * c["brightness"] * strength
    return x + signed(3, 1, 1) * c["channel_shift"] * strength


_warned_no_camera_px = False


def _camera_ratios(batch: int, cfg: dict, strength: float, fit: torch.Tensor) -> list[float] | None:
    """Per sample, camera resolution as a fraction of the frame resolution. None if the step does not apply.

    Each sample draws its camera pixels per tile log-uniformly from
    ``camera_px_per_tile``. The frame shows a tile at ``fit * TILE_PX`` pixels,
    so the fraction is the drawn value over that. A fraction of 1 or more
    means the frame is already no finer than the camera and nothing is lost.
    """
    global _warned_no_camera_px
    span = camera_range(cfg)
    if span is None:
        if not _warned_no_camera_px:
            warnings.warn(
                "camera_px_per_tile is unset: the loss-of-resolution step is skipped. "
                "Every other simulation step runs.",
                stacklevel=2,
            )
            _warned_no_camera_px = True
        return None
    low, high = span
    camera = low * (high / low) ** torch.rand(batch, dtype=torch.float64)
    ratio = camera / (TILE_PX * fit.double().cpu())
    ratio = 1.0 + strength * (ratio - 1.0)          # strength 0: no loss
    return ratio.tolist()


def _sensor(frame: torch.Tensor, sigma: torch.Tensor | None, quality: torch.Tensor | None, chroma: bool) -> torch.Tensor:
    """Noise, clipping to [0,1] and the approximate JPEG, at whatever resolution ``frame`` has."""
    if sigma is not None:
        frame = frame + torch.randn_like(frame) * sigma.view(-1, 1, 1, 1)
    frame = frame.clamp(0.0, 1.0)
    if quality is not None:
        frame = jpeg_approx(frame, quality, chroma).clamp(0.0, 1.0)
    return frame


# -- the simulation ---------------------------------------------------------


def simulate(surface: torch.Tensor, cfg: dict, strength: float, direct: bool = False) -> SimResult:
    """Wrapped surface -> (camera frame, surface-to-frame homography).

    ``strength`` is the single knob, 0 to 1. ``direct`` adds the random
    rotation, scale and shift used when the reader gets the frame unflattened;
    those apply whenever the simulation is enabled, at any strength, unless
    ``simulation.direct.ramp_with_strength`` is set.

    With the simulation disabled the frame is the surface and the homography
    is the identity. The same holds, without any resampling, in flattened
    mode when no perspective applies.
    """
    sim = cfg["simulation"]
    b, _, surface_px, _ = surface.shape
    identity = torch.eye(3, device=surface.device, dtype=surface.dtype).repeat(b, 1, 1)
    if not sim["enabled"]:
        return SimResult(surface, identity)
    effects = sim["effects"]
    on = strength > 0

    if direct or (effects["perspective"] and on):
        h64, fit = sample_homography(b, surface_px, cfg, strength, direct)
        frame_w, frame_h = frame_size(cfg)
        frame = warp(surface, torch.linalg.inv(h64).to(surface), (frame_h, frame_w))
        homography = h64.to(surface)
    else:
        frame, homography, fit = surface, identity, torch.ones(b, dtype=torch.float64)
    if not on:
        return SimResult(frame, homography)

    def uniform(scale: float) -> torch.Tensor:
        return torch.rand(b, device=surface.device, dtype=surface.dtype) * scale * strength

    if effects["blur"]:
        kernels = blur_kernels(
            uniform(sim["blur"]["defocus_radius_px"]),
            uniform(sim["blur"]["motion_length_px"]),
            torch.rand(b, device=surface.device, dtype=surface.dtype) * math.pi,
            _blur_kernel_size(sim["blur"]),
        )
        frame = apply_blur(frame, kernels)

    if effects["colour"]:
        frame = colour_shift(frame, sim["colour"], strength)

    sigma = uniform(sim["noise"]["sigma"]) if effects["noise"] else None
    quality = 100.0 - uniform(100.0 - sim["jpeg"]["min_quality"]) if effects["jpeg"] else None
    chroma = sim["jpeg"]["chroma_subsampling"]
    ratios = _camera_ratios(b, cfg, strength, fit) if effects["resolution"] else None
    if ratios is None:
        return SimResult(_sensor(frame, sigma, quality, chroma), homography)

    # Each sample has its own camera resolution, so each goes down, through
    # the sensor and back up on its own.
    full_h, full_w = frame.shape[-2:]
    out = []
    for i, ratio in enumerate(ratios):
        one = frame[i : i + 1]
        lossy = ratio < 1.0
        if lossy:
            size = (max(16, round(full_h * ratio)), max(16, round(full_w * ratio)))
            one = F.interpolate(one, size=size, mode="bilinear", antialias=True, align_corners=False)
        one = _sensor(
            one,
            None if sigma is None else sigma[i : i + 1],
            None if quality is None else quality[i : i + 1],
            chroma,
        )
        if lossy:
            one = F.interpolate(one, size=(full_h, full_w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        out.append(one)
    return SimResult(torch.cat(out, dim=0), homography)
