"""How visible the mark is on the printed tile: S-CIELAB (Zhang and Wandell, 1996) with masking.

The clean and marked tiles, sRGB in [0, 1], go to linear light, CIE XYZ and
the opponent space of S-CIELAB (one luminance and two colour channels). Each
channel is blurred the way the eye blurs it at the viewing distance, then the
result goes back to XYZ and CIELAB, and the per-pixel colour difference delta E (CIE76)
between the two is the visibility map. About 1 is the threshold of a visible
difference (only roughly, for print).

The blur is a circular FFT over the tile: the print repeats the tile, so
circular filtering is exact. The map is then multiplied by a masking factor
from the clean tile: 1 on flat colour, lower on text, edges and texture,
where the eye notices a change less.
"""

from __future__ import annotations

import math

import torch

from constants import TILE_PX
from losses import texture_settings, texture_weights

# Used when the config has no loss.visibility.
VISIBILITY_DEFAULTS = {"enabled": False, "distance_mm": 500, "masking_floor": None, "hinge": False}

# sRGB (IEC 61966-2-1, D65) linear light to CIE XYZ. White (1, 1, 1) maps to WHITE below.
SRGB_TO_XYZ = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
WHITE = (0.9505, 1.0, 1.089)

# XYZ to the opponent channels of S-CIELAB (Poirson and Wandell): luminance, red-green, blue-yellow.
XYZ_TO_OPPONENT = [[0.279, 0.72, -0.107], [-0.449, 0.29, -0.077], [0.086, -0.59, 0.501]]
OPPONENT_TO_XYZ = torch.linalg.inv(torch.tensor(XYZ_TO_OPPONENT, dtype=torch.float64)).tolist()
OPPONENT_TO_LINEAR = (torch.linalg.inv(torch.tensor(SRGB_TO_XYZ, dtype=torch.float64)) @ torch.tensor(OPPONENT_TO_XYZ, dtype=torch.float64)).tolist()

# S-CIELAB spatial filters (Zhang and Wandell), one per opponent channel: a weighted sum of Gaussians
# exp(-(x^2 + y^2) / s^2), each normalised to sum 1, as (weight, s in degrees of visual angle). The
# weights of each channel sum to 1, so flat colour passes unchanged. The spreads are the paper's, for
# this form of the Gaussian; the reference scielab code lists 0.05, 0.225 and 7.0 degrees and so on,
# but those are full widths at half maximum, about 1.67 times these.
FILTERS = (
    ((1.00327, 0.0283), (0.114416, 0.133), (-0.117686, 4.336)),  # luminance
    ((0.616725, 0.0392), (0.383275, 0.494)),  # red-green
    ((0.567885, 0.0536), (0.432115, 0.386)),  # blue-yellow
)

DELTA_E_EPS = 1e-8  # under the square root of delta E, so its gradient stays finite where nothing changed

_spectra_cache: dict[tuple, torch.Tensor] = {}


def pixels_per_degree(tile_mm: float, distance_mm: float, tile_px: int = TILE_PX) -> float:
    """Tile pixels per degree of visual angle, for a printed tile ``tile_mm`` wide seen from ``distance_mm``."""
    degrees = math.degrees(2.0 * math.atan(0.5 * float(tile_mm) / float(distance_mm)))
    return float(tile_px) / degrees


def visibility_settings(cfg: dict) -> dict:
    """``loss.visibility`` merged over the defaults, checked."""
    s = {**VISIBILITY_DEFAULTS, **(cfg.get("loss", {}).get("visibility") or {})}
    if float(s["distance_mm"]) <= 0.0:
        raise ValueError(f"loss.visibility.distance_mm must be above 0, got {s['distance_mm']}")
    floor = s["masking_floor"]
    if floor is not None and not 0.0 < float(floor) <= 1.0:
        raise ValueError(f"loss.visibility.masking_floor must be null or in (0, 1], got {floor}")
    if not isinstance(s["hinge"], bool):
        raise ValueError(f"loss.visibility.hinge must be true or false, got {s['hinge']!r}")
    return s


def _periodic_gaussian(n: int, spread_px: float) -> torch.Tensor:
    """exp(-x^2 / s^2) on a circle of n pixels, (n,) float64: summed over the tile's repeats, so a spread
    wider than the tile is still exact on the repeating print. For a narrow spread it is the Gaussian
    of the wrapped distance."""
    x = torch.arange(n, dtype=torch.float64)
    repeats = math.ceil(5.0 * spread_px / n) + 1
    g = torch.zeros(n, dtype=torch.float64)
    for k in range(-repeats, repeats + 1):
        g += torch.exp(-((x + k * n) / spread_px) ** 2)
    return g


def filter_spectra(height: int, width: int, ppd: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The three S-CIELAB filters as FFT spectra on a height x width tile, (3, H, W), real.

    Each Gaussian is built on the tile grid, normalised to sum 1 and weighted; the value at frequency
    (0, 0) is the filter's DC gain, 1. Cached per (size, ppd, device, dtype).
    """
    key = (height, width, float(ppd), str(torch.device(device)), dtype)
    if key not in _spectra_cache:
        kernels = []
        for channel in FILTERS:
            kernel = torch.zeros(height, width, dtype=torch.float64)
            for weight, spread_deg in channel:
                spread_px = spread_deg * ppd
                g = torch.outer(_periodic_gaussian(height, spread_px), _periodic_gaussian(width, spread_px))
                kernel += weight * g / g.sum()
            kernels.append(kernel)
        spectra = torch.fft.fft2(torch.stack(kernels)).real  # the kernels are symmetric, so the spectra are real
        _spectra_cache[key] = spectra.to(device=device, dtype=dtype)
    return _spectra_cache[key]


def _mix(matrix: list[list[float]], image: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 colour matrix to the channels of (B,3,H,W)."""
    m = torch.as_tensor(matrix, dtype=image.dtype, device=image.device)
    return torch.einsum("ij,bjhw->bihw", m, image)


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """IEC 61966-2-1. Values outside [0, 1] continue the curve (linear below 0, the power law above 1)."""
    curve = ((rgb.clamp(min=0.04045) + 0.055) / 1.055).pow(2.4)
    return torch.where(rgb <= 0.04045, rgb / 12.92, curve)


def _linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    curve = 1.055 * linear.clamp(min=0.0031308).pow(1.0 / 2.4) - 0.055
    return torch.where(linear <= 0.0031308, linear * 12.92, curve)


def _working(image: torch.Tensor) -> torch.Tensor:
    """float32 at least: half precision is not enough for the colour maths and the FFT."""
    return image if image.dtype in (torch.float32, torch.float64) else image.float()


def opponent(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB (B,3,H,W) to the linear-light opponent channels of S-CIELAB: luminance, red-green, blue-yellow."""
    return _mix(XYZ_TO_OPPONENT, _mix(SRGB_TO_XYZ, _srgb_to_linear(_working(rgb))))


def from_opponent(opp: torch.Tensor) -> torch.Tensor:
    """Opponent channels (B,3,H,W) back to sRGB; undoes ``opponent``."""
    return _linear_to_srgb(_mix(OPPONENT_TO_LINEAR, opp))


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """CIE XYZ (B,3,H,W) to CIELAB against WHITE, with the standard linear segment below (6/29)^3."""
    white = torch.tensor(WHITE, dtype=xyz.dtype, device=xyz.device).view(1, 3, 1, 1)
    t = xyz / white
    delta = 6.0 / 29.0
    # The cube root only sees values where it is used, so neither branch can give a NaN gradient.
    f = torch.where(t > delta**3, t.clamp(min=delta**3).pow(1.0 / 3.0), t / (3.0 * delta**2) + 4.0 / 29.0)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def _scielab(rgb: torch.Tensor, ppd: float) -> torch.Tensor:
    """sRGB (B,3,H,W) to CIELAB after the S-CIELAB blur."""
    opp = opponent(rgb)
    spectra = filter_spectra(opp.shape[-2], opp.shape[-1], ppd, opp.device, opp.dtype)
    blurred = torch.fft.ifft2(torch.fft.fft2(opp) * spectra).real
    return _xyz_to_lab(_mix(OPPONENT_TO_XYZ, blurred))


def delta_e_map(clean: torch.Tensor, marked: torch.Tensor, ppd: float) -> torch.Tensor:
    """S-CIELAB delta E (CIE76) per pixel between the clean and marked tiles, (B,1,H,W), without masking.

    sqrt(dL^2 + da^2 + db^2 + eps) - sqrt(eps): exactly 0 where nothing changed, with a finite gradient there.
    """
    assert clean.shape == marked.shape, f"clean {tuple(clean.shape)} and marked {tuple(marked.shape)} tiles must have the same shape"
    if clean.requires_grad:
        lab = _scielab(torch.cat([clean, marked], dim=0), ppd)
        clean_lab, marked_lab = lab.chunk(2, dim=0)
    else:
        # The clean half needs no gradient: keep it out of the autograd graph, which halves the backward memory.
        with torch.no_grad():
            clean_lab = _scielab(clean, ppd)
        marked_lab = _scielab(marked, ppd)
    squared = (marked_lab - clean_lab).pow(2).sum(dim=1, keepdim=True)
    return (squared + DELTA_E_EPS).sqrt() - math.sqrt(DELTA_E_EPS)


@torch.no_grad()
def masking_factor(clean: torch.Tensor, cfg: dict) -> torch.Tensor:
    """How much less the eye notices a change at each pixel, (B,1,H,W) in [floor, 1], from the clean tile.

    sqrt(texture_weights / flat_weight) with the texture mask on, whatever the config says: 1 on flat
    colour, sqrt(busy_weight / flat_weight) on busy detail. ``loss.visibility.masking_floor`` can raise
    the floor; null keeps sqrt(busy_weight / flat_weight).
    """
    loss = cfg.get("loss") or {}
    mask_cfg = {**cfg, "loss": {**loss, "texture_mask": {**texture_settings(cfg), "enabled": True}}}
    t = texture_settings(mask_cfg)
    flat, busy = float(t["flat_weight"]), float(t["busy_weight"])
    floor = visibility_settings(cfg)["masking_floor"]
    floor = math.sqrt(busy / flat) if floor is None else float(floor)
    return (texture_weights(_working(clean), mask_cfg) / flat).sqrt().clamp(min=floor, max=1.0)


def _row_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile over each row of (B, N), in chunks: torch.quantile refuses inputs over 2^24 elements."""
    rows = max(1, (1 << 24) // values.shape[1])
    return torch.cat([torch.quantile(chunk, q, dim=1) for chunk in values.split(rows, dim=0)])


def tile_visibility(clean: torch.Tensor, marked: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    """The masked S-CIELAB delta E of the mark on each tile.

    ``map`` (B,1,H,W) per pixel; ``mean`` (B,) per tile, the loss; ``p95`` (B,) the 95th percentile over
    the tile's pixels, the tile's visibility. The tile is ``demo.tile_mm`` wide, seen from
    ``loss.visibility.distance_mm``.
    """
    s = visibility_settings(cfg)
    tile_mm = float((cfg.get("demo") or {}).get("tile_mm", 50))
    ppd = pixels_per_degree(tile_mm, float(s["distance_mm"]), clean.shape[-1])
    masked = delta_e_map(clean, marked, ppd) * masking_factor(clean, cfg)
    return {"map": masked, "mean": masked.mean(dim=(1, 2, 3)), "p95": _row_quantile(masked.detach().flatten(1), 0.95)}


def batch_statistic(p95: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    """The batch's visibility for the controller: a high quantile over tiles, so flat tiles cannot hide in an average."""
    return torch.quantile(p95.float(), quantile)
