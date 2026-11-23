"""Differentiable print-and-camera simulation between the embedder and the reader.

``simulate`` takes the wrapped surface (the marked tile repeated) and returns
the camera frame together with the exact homography it applied, mapping
surface pixels to frame pixels (coordinates as in geometry.py).

One knob, ``strength`` in [0, 1], scales every range. Order of the steps:

1. perspective warp           tilt up to max_tilt_deg * strength
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

Direct mode (the reader gets the camera frame unflattened) is its own path,
``_simulate_direct``. There the frame is in camera pixels: the tile is drawn
at the size the camera shows it, 154 to 340 pixels with the phone settings,
at a random rotation and position, and the frame is the TILE_PX square the
reader is given. Steps 4 and 7 do not exist in that path; drawing the tile at
camera size is the loss of resolution.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import torch
import torch.nn.functional as F

from camera import camera_range, frame_size
from constants import TILE_PX
from geometry import apply_homography, warp, warp_periodic


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


def _random_tilt(batch: int, cfg: dict, strength: float) -> torch.Tensor | None:
    """A random tilt of the surface, acting on centred surface coordinates, or None when it is off."""
    sim = cfg["simulation"]
    if not (sim["effects"]["perspective"] and strength > 0):
        return None
    p = sim["perspective"]
    tilt = torch.rand(batch, dtype=torch.float64) * math.radians(p["max_tilt_deg"]) * strength
    azimuth = torch.rand(batch, dtype=torch.float64) * 2 * math.pi
    lo, hi = p["distance_tiles"]
    distance = (lo + torch.rand(batch, dtype=torch.float64) * (hi - lo)) * TILE_PX
    return _tilt(tilt, azimuth, distance)


def _rotation(batch: int, cfg: dict, strength: float) -> torch.Tensor | None:
    """In-plane rotation of the surface, uniform in [0, rotation_deg * strength]; None when there is none."""
    sim = cfg["simulation"]
    deg = float(sim["perspective"].get("rotation_deg") or 0.0)
    if not (sim["effects"]["perspective"] and strength > 0 and deg > 0):
        return None
    angle = torch.rand(batch, dtype=torch.float64) * math.radians(deg) * strength
    return _scale_rotate(torch.ones(batch, dtype=torch.float64), angle)


def tile_px_in_frame(cfg: dict) -> tuple[float, float] | None:
    """``simulation.perspective.tile_px_in_frame``: (low, high) frame pixels per tile, or None (fit to the frame).

    ``auto`` is for the locator: ``camera_px_per_tile`` shrunk as the app
    shrinks the central square of the stream to the locator input
    (``locator_input_scale``).
    """
    value = cfg["simulation"]["perspective"].get("tile_px_in_frame")
    if value is None:
        return None
    if value == "auto":
        span = camera_range(cfg)
        if span is None:
            raise ValueError("simulation.perspective.tile_px_in_frame: auto needs camera_px_per_tile")
        scale = locator_input_scale(cfg)
        return span[0] * scale, span[1] * scale
    low, high = float(value[0]), float(value[1])
    if not 0 < low <= high:
        raise ValueError(f"simulation.perspective.tile_px_in_frame needs 0 < low <= high, got {value!r}")
    return low, high


def locator_crop_px(cfg: dict) -> int:
    """Side of the central square of the camera stream the app gives the locator: the stream's short side."""
    stream = (cfg.get("demo") or {}).get("stream_px")
    if not stream:
        raise ValueError("the locator needs demo.stream_px, the camera stream size [width, height]")
    return int(min(stream))


def locator_input_scale(cfg: dict) -> float:
    """Locator-input pixels per camera pixel: locator input side over the crop side (384 / 1080)."""
    return float(cfg["locator_train"]["input_px"]) / locator_crop_px(cfg)


def sample_homography(
    batch: int,
    surface_px: int,
    cfg: dict,
    strength: float,
    centre: torch.Tensor | None = None,
    tile_px: tuple[float, float] | None = None,
    frame_wh: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened mode: random surface -> frame homography, (B,3,3) float64 on the CPU, and the scale (B,).

    The surface point ``centre`` (B,2), by default the middle of the
    surface, lands on the middle of the frame. Order: in-plane rotation
    (``simulation.perspective.rotation_deg``), tilt, scale.

    The scale is the frame's pixels per surface pixel near the middle. With
    ``tile_px`` (or ``simulation.perspective.tile_px_in_frame``) unset it is
    the fit scale, how much each sample was shrunk to fit the frame; that is
    not part of the camera model, so the loss-of-resolution step divides it
    out again. With it set, the tile is drawn log-uniformly at that many
    frame pixels and nothing is fitted.
    """
    frame_w, frame_h = frame_wh or frame_size(cfg)
    if centre is None:
        centre = torch.full((batch, 2), surface_px / 2.0, dtype=torch.float64)
    centre = centre.to(torch.float64)
    h = _translate(-centre[:, 0], -centre[:, 1])
    rotation = _rotation(batch, cfg, strength)
    if rotation is not None:
        h = rotation @ h
    tilt = _random_tilt(batch, cfg, strength)
    if tilt is not None:
        h = tilt @ h

    span = tile_px if tile_px is not None else tile_px_in_frame(cfg)
    if span is None:
        # Keep the middle two tiles' worth of surface inside the frame, so a
        # tile-sized patch at any offset can be flattened out of it.
        reach = TILE_PX + 8.0
        corners = torch.tensor([[-reach, -reach], [reach, -reach], [reach, reach], [-reach, reach]], dtype=torch.float64)
        projected = apply_homography(h, corners.unsqueeze(0) + centre.unsqueeze(1))
        extent_x, extent_y = projected[..., 0].abs().amax(dim=1), projected[..., 1].abs().amax(dim=1)
        scale = torch.minimum((frame_w / 2.0 - 4.0) / extent_x, (frame_h / 2.0 - 4.0) / extent_y).clamp(max=1.0)
    else:
        low, high = span
        scale = low * (high / low) ** torch.rand(batch, dtype=torch.float64) / TILE_PX
    h = _scale_rotate(scale, torch.zeros_like(scale)) @ h

    middle_x = torch.full((batch,), frame_w / 2.0, dtype=torch.float64)
    middle_y = torch.full((batch,), frame_h / 2.0, dtype=torch.float64)
    return _translate(middle_x, middle_y) @ h, scale


def direct_scale_range(cfg: dict) -> tuple[float, float]:
    """Direct mode: the size of a tile in the frame, as a fraction of TILE_PX, (low, high).

    ``simulation.direct.scale: auto`` takes it from ``camera_px_per_tile``,
    which is the point of direct mode: the reader sees the tile at the size
    the camera shows it. ``[low, high]`` sets it by hand.
    """
    value = cfg["simulation"]["direct"]["scale"]
    if value == "auto":
        span = camera_range(cfg)
        if span is None:
            raise ValueError("simulation.direct.scale: auto needs camera_px_per_tile; set that or give [low, high]")
        return span[0] / TILE_PX, span[1] / TILE_PX
    low, high = float(value[0]), float(value[1])
    if not 0 < low <= high:
        raise ValueError(f"simulation.direct.scale needs 0 < low <= high, got {value!r}")
    return low, high


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


def _simulate_direct(surface: torch.Tensor, cfg: dict, strength: float) -> SimResult:
    """Direct mode: the TILE_PX square of camera frame the reader is given.

    Per sample: the surface, taken as repeating without end, is drawn at full
    resolution with a random position, tilt and rotation into a square just
    large enough; blur and colour are applied there; the square is then
    reduced, with averaging as a sensor does, to TILE_PX. That last step puts
    the tile at its camera size, ``direct_scale_range`` times TILE_PX. Noise
    and JPEG follow at that resolution.

    Rotation and size apply at any strength unless
    ``simulation.direct.ramp_with_strength`` is set; then both ranges grow
    with the strength. ``simulation.direct.rotation_near_right_angles_deg``
    replaces the free rotation by one of the four right-angle orientations
    plus or minus that many degrees.
    """
    sim = cfg["simulation"]
    effects, d = sim["effects"], sim["direct"]
    b, _, surface_px, _ = surface.shape
    on = strength > 0
    amount = strength if d["ramp_with_strength"] else 1.0

    half = torch.full((b,), -surface_px / 2.0, dtype=torch.float64)
    # Random position within the tile pattern, so the frame starts anywhere in a tile.
    shift = (torch.rand(b, 2, dtype=torch.float64) - 0.5) * TILE_PX
    h = _translate(shift[:, 0], shift[:, 1]) @ _translate(half, half)
    tilt = _random_tilt(b, cfg, strength)
    if tilt is not None:
        h = tilt @ h
    near = d.get("rotation_near_right_angles_deg")
    if near is None:
        angle = torch.rand(b, dtype=torch.float64) * math.radians(d["rotation_deg"]) * amount
    else:
        # Upright, on either side or upside down, each within plus or minus `near` degrees.
        quarter = torch.randint(0, 4, (b,)).double() * (math.pi / 2)
        angle = quarter + (torch.rand(b, dtype=torch.float64) * 2 - 1) * math.radians(near) * amount
    h = _scale_rotate(torch.ones(b, dtype=torch.float64), angle) @ h
    low, high = direct_scale_range(cfg)
    log_scale = math.log(low) + torch.rand(b, dtype=torch.float64) * (math.log(high) - math.log(low))
    # Side of the full-resolution square that becomes the TILE_PX frame.
    sides = [max(32, round(TILE_PX / s)) for s in torch.exp(log_scale * amount).tolist()]

    def uniform(scale: float) -> torch.Tensor:
        return torch.rand(b, device=surface.device, dtype=surface.dtype) * scale * strength

    kernels = None
    if on and effects["blur"]:
        kernels = blur_kernels(
            uniform(sim["blur"]["defocus_radius_px"]),
            uniform(sim["blur"]["motion_length_px"]),
            torch.rand(b, device=surface.device, dtype=surface.dtype) * math.pi,
            _blur_kernel_size(sim["blur"]),
        )

    frames, homographies = [], []
    for i, side in enumerate(sides):
        middle = torch.full((1,), side / 2.0, dtype=torch.float64)
        to_square = _translate(middle, middle) @ h[i : i + 1]
        view = warp_periodic(surface[i : i + 1], torch.linalg.inv(to_square).to(surface), (side, side))
        if kernels is not None:
            view = apply_blur(view, kernels[i : i + 1])
        if on and effects["colour"]:
            view = colour_shift(view, sim["colour"], strength)
        if side != TILE_PX:
            view = F.interpolate(view, size=(TILE_PX, TILE_PX), mode="bilinear", antialias=True, align_corners=False)
        frames.append(view)
        to_frame = torch.full((1,), TILE_PX / side, dtype=torch.float64)
        homographies.append(_scale_rotate(to_frame, torch.zeros(1, dtype=torch.float64)) @ to_square)
    frame, homography = torch.cat(frames, dim=0), torch.cat(homographies, dim=0).to(surface)
    if not on:
        return SimResult(frame, homography)
    sigma = uniform(sim["noise"]["sigma"]) if effects["noise"] else None
    quality = 100.0 - uniform(100.0 - sim["jpeg"]["min_quality"]) if effects["jpeg"] else None
    return SimResult(_sensor(frame, sigma, quality, sim["jpeg"]["chroma_subsampling"]), homography)


def simulate(surface: torch.Tensor, cfg: dict, strength: float, direct: bool = False) -> SimResult:
    """Surface -> (camera frame, surface-to-frame homography).

    ``strength`` is the single knob, 0 to 1. ``direct`` is the mode in which
    the reader gets the frame unflattened; see ``_simulate_direct``. There the
    surface is taken as periodic, so one tile is enough, and the frame is
    TILE_PX square.

    With the simulation disabled the frame is the surface and the homography
    is the identity. The same holds, without any resampling, in flattened
    mode when no perspective applies.
    """
    sim = cfg["simulation"]
    b, _, surface_px, _ = surface.shape
    identity = torch.eye(3, device=surface.device, dtype=surface.dtype).repeat(b, 1, 1)
    if not sim["enabled"]:
        return SimResult(surface, identity)
    if direct:
        return _simulate_direct(surface, cfg, strength)
    effects = sim["effects"]
    on = strength > 0

    if effects["perspective"] and on:
        h64, fit = sample_homography(b, surface_px, cfg, strength)
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


# -- the locator's view -----------------------------------------------------


class ViewResult(NamedTuple):
    frame: torch.Tensor        # (B,3,H,W) in [0,1]
    homography: torch.Tensor   # (B,3,3): surface pixels -> frame pixels, exact; the surface is one tile, repeating
    corners: torch.Tensor      # (B,4,2): the locator's target in frame pixels, see target_corners
    tile_px: torch.Tensor      # (B,): pixels per tile in the frame near its middle (the drawn tile size)


def surface_extent_tiles(cfg: dict) -> tuple[float, float] | None:
    """``simulation.surface_extent_tiles``: (low, high) tiles from the centre point to each edge of the print, or None."""
    value = cfg["simulation"].get("surface_extent_tiles")
    if value is None:
        return None
    low, high = float(value[0]), float(value[1])
    if not 0.5 < low <= high:
        raise ValueError(
            f"simulation.surface_extent_tiles needs 0.5 < low <= high (the target square reaches half a tile), got {value!r}"
        )
    return low, high


def target_corners(homography: torch.Tensor, frame_w: int, frame_h: int) -> torch.Tensor:
    """The locator's target: the tile-sized square on the surface centred on the point seen at the middle of the frame.

    Sides along the surface rows and columns, carried into the frame by
    ``homography`` (surface -> frame). Order top-left, top-right,
    bottom-right, bottom-left as on the surface. (B,4,2).
    """
    h = homography.double().cpu()
    b = h.shape[0]
    middle = torch.tensor([[[frame_w / 2.0, frame_h / 2.0]]], dtype=torch.float64).expand(b, -1, -1)
    centre = apply_homography(torch.linalg.inv(h), middle)
    half = TILE_PX / 2.0
    offsets = torch.tensor([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=torch.float64)
    return apply_homography(h, centre + offsets.unsqueeze(0)).to(homography)


def _pixel_centres(h: int, w: int, b: int, device, dtype) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=dtype) + 0.5
    xs = torch.arange(w, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(1, -1, 2).expand(b, -1, -1)


def _background(background: torch.Tensor, frame_hw: tuple[int, int]) -> torch.Tensor:
    """Per sample, a flat colour or a zoomed, turned window of another label tile, filling the frame. Half each."""
    b = background.shape[0]
    fh, fw = frame_hw
    device, dtype = background.device, background.dtype
    # Frame pixels -> background tile pixels: a window 0.25 to 0.75 tiles wide, any angle, anywhere.
    side = (0.25 + 0.5 * torch.rand(b, dtype=torch.float64)) * TILE_PX
    scale = side / max(fw, fh)
    angle = torch.rand(b, dtype=torch.float64) * 2 * math.pi
    where = torch.rand(b, 2, dtype=torch.float64) * TILE_PX
    to_tile = (
        _translate(where[:, 0], where[:, 1])
        @ _scale_rotate(scale, angle)
        @ _translate(torch.full((b,), -fw / 2.0, dtype=torch.float64), torch.full((b,), -fh / 2.0, dtype=torch.float64))
    )
    picture = warp_periodic(background, to_tile.to(device=device, dtype=dtype), (fh, fw))
    flat = torch.rand(b, 3, 1, 1, device=device, dtype=dtype).expand(-1, -1, fh, fw)
    use_flat = (torch.rand(b, device=device) < 0.5).view(b, 1, 1, 1)
    return torch.where(use_flat, flat, picture)


def _warp_shrunk(tile: torch.Tensor, to_surface: torch.Tensor, scale: torch.Tensor, frame_hw: tuple[int, int]) -> torch.Tensor:
    """``warp_periodic`` of the repeating tile, shrunk first with smoothing where the frame is coarser than the tile.

    Bilinear sampling alone would alias at 3 to 7 times smaller. Each sample
    whose scale is below 0.8 gets its tile resized, with antialiasing, to
    about 1.25 times its size in the frame, and is sampled from that.
    """
    device, dtype = tile.device, tile.dtype
    out = []
    for i in range(tile.shape[0]):
        one, inv = tile[i : i + 1], to_surface[i : i + 1]
        s = float(scale[i])
        if s < 0.8:
            side = max(16, round(TILE_PX * s * 1.25))
            one = F.interpolate(one, size=(side, side), mode="bilinear", antialias=True, align_corners=False)
            k = torch.full((1,), side / TILE_PX, dtype=torch.float64)
            inv = _scale_rotate(k, torch.zeros(1, dtype=torch.float64)) @ inv
        out.append(warp_periodic(one, inv.to(device=device, dtype=dtype), frame_hw))
    return torch.cat(out, dim=0)


def simulate_view(
    tile: torch.Tensor,
    cfg: dict,
    strength: float,
    frame_wh: tuple[int, int],
    tile_px: tuple[float, float],
    background: torch.Tensor | None = None,
    max_tries: int = 20,
) -> ViewResult:
    """The locator's view: one tile, repeating, seen at ``tile_px`` frame pixels per tile, and its target corners.

    The tile is taken as an endless pattern and the point at the middle of
    the frame is a random place in it. Geometry from ``sample_homography``
    (rotation, tilt, the tile size drawn from ``tile_px``). With
    ``simulation.surface_extent_tiles`` set the print ends: each of its four
    edges lies that many tiles (drawn per edge) from the centre point, and
    beyond it the frame shows ``background`` (see ``_background``); the
    target square is always inside the print. A sample whose target square
    leaves the frame is drawn again.

    Then blur (its lengths, given at TILE_PX pixels per tile, scaled to the
    drawn tile size), colour, noise and JPEG at the resolution of the frame.
    No loss-of-resolution step: the caller picks the resolution.
    """
    sim = cfg["simulation"]
    effects = sim["effects"]
    b = tile.shape[0]
    fw, fh = frame_wh
    device, dtype = tile.device, tile.dtype
    on = strength > 0

    centre = torch.rand(b, 2, dtype=torch.float64) * TILE_PX
    h, scale = sample_homography(b, TILE_PX, cfg, strength, centre=centre, tile_px=tile_px, frame_wh=frame_wh)
    corners = target_corners(h, fw, fh)
    for _ in range(max_tries):
        bad = (
            (corners[..., 0].amin(1) < 1) | (corners[..., 0].amax(1) > fw - 1)
            | (corners[..., 1].amin(1) < 1) | (corners[..., 1].amax(1) > fh - 1)
        )
        if not bool(bad.any()):
            break
        idx = bad.nonzero().flatten()
        h2, s2 = sample_homography(len(idx), TILE_PX, cfg, strength, centre=centre[idx], tile_px=tile_px, frame_wh=frame_wh)
        h[idx], scale[idx] = h2, s2
        corners = target_corners(h, fw, fh)
    else:
        raise RuntimeError(f"simulate_view: no target square inside a {fw}x{fh} frame after {max_tries} draws")

    to_surface = torch.linalg.inv(h)
    frame = _warp_shrunk(tile, to_surface, scale, (fh, fw))
    extent = surface_extent_tiles(cfg)
    if extent is not None:
        if background is None:
            raise ValueError("simulation.surface_extent_tiles is set: simulate_view needs a background")
        low, high = extent
        reach = (low + (high - low) * torch.rand(b, 4, dtype=torch.float64)) * TILE_PX   # left, top, right, bottom
        src = apply_homography(to_surface.to(device=device, dtype=dtype), _pixel_centres(fh, fw, b, device, dtype))
        rel = src - centre.to(device=device, dtype=dtype).unsqueeze(1)
        reach = reach.to(device=device, dtype=dtype)
        inside = torch.minimum(
            torch.minimum(rel[..., 0] + reach[:, 0:1], reach[:, 2:3] - rel[..., 0]),
            torch.minimum(rel[..., 1] + reach[:, 1:2], reach[:, 3:4] - rel[..., 1]),
        )
        # About one frame pixel of soft edge, as a sampled print edge has.
        mask = (inside * scale.to(device=device, dtype=dtype).view(b, 1) + 0.5).clamp(0.0, 1.0).view(b, 1, fh, fw)
        frame = mask * frame + (1.0 - mask) * _background(background, (fh, fw))

    homography = h.to(device=device, dtype=dtype)
    corners = corners.to(device=device, dtype=dtype)
    tile_size = (scale * TILE_PX).to(device=device, dtype=dtype)
    if not on:
        return ViewResult(frame, homography, corners, tile_size)

    def uniform(value: float) -> torch.Tensor:
        return torch.rand(b, device=device, dtype=dtype) * value * strength

    if effects["blur"]:
        k = scale.to(device=device, dtype=dtype)
        kernels = blur_kernels(
            uniform(sim["blur"]["defocus_radius_px"]) * k,
            uniform(sim["blur"]["motion_length_px"]) * k,
            torch.rand(b, device=device, dtype=dtype) * math.pi,
            _blur_kernel_size(sim["blur"]),
        )
        frame = apply_blur(frame, kernels)
    if effects["colour"]:
        frame = colour_shift(frame, sim["colour"], strength)
    sigma = uniform(sim["noise"]["sigma"]) if effects["noise"] else None
    quality = 100.0 - uniform(100.0 - sim["jpeg"]["min_quality"]) if effects["jpeg"] else None
    frame = _sensor(frame, sigma, quality, sim["jpeg"]["chroma_subsampling"])
    return ViewResult(frame, homography, corners, tile_size)
