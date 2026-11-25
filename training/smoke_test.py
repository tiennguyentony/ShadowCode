"""Smoke test: 300 stage 1 training steps on a tiny generated dataset, simulation off.

Asserts that bit accuracy rises well above chance (50 percent). Runs on CPU
or MPS, and uses a CUDA GPU when there is one. Uses the real models and the
real training step: embed, repeat 3x3, flatten one tile-sized patch at a
random offset with corner jitter, read.

Also reports the PSNR between marked and original tiles and saves a sheet
with original, marked and residual x10 side by side.

    python smoke_test.py                 # fastest device available
    python smoke_test.py --device cpu
    python smoke_test.py --config configs/smoke_blur.yaml band.grating_step=4
    python smoke_test.py data.source=generated_labels    # on labelgen.py tiles
    pytest smoke_test.py -s
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from checkpoint import load_checkpoint, save_checkpoint
from config import load_config
from constants import MESSAGE_BITS, TILE_PX
from torchvision import transforms

from data import ImageFolderDataset, holdout_images, infinite, load_rgb, make_loader, procedural_image
from losses import texture_settings, texture_weights
from geometry import apply_homography, flatten_from_corners, flatten_random_patch, square_corners, warp_periodic
from models import count_parameters
from simulate import simulate
from visibility import batch_statistic, from_opponent, opponent, tile_visibility, visibility_settings
from spread import concentration, spread_report, hide_mask, hide_regions, hide_settings, local_energy, peak_penalty, peak_settings
from train import Trainer
from viz import sample_sheet, save_image

HERE = Path(__file__).resolve().parent
REQUIRED_ACCURACY = 0.70  # chance is 0.50
READER_PARAM_LIMIT = 3_000_000
NUM_IMAGES = 24


def make_tiny_dataset(root: Path, count: int = NUM_IMAGES, size: int = 448, seed: int = 0) -> None:
    """Write procedural images (smooth colour fields, shapes, grain) to root."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        Image.fromarray(procedural_image(rng, size)).save(root / f"img_{i:03d}.png")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_structure(trainer: Trainer) -> None:
    """Cheap checks of the fixed parts of the spec, before any training."""
    for name, module in trainer.embedder.named_modules():
        if isinstance(module, nn.Conv2d):
            assert module.padding_mode == "circular", f"embedder conv {name} is not circular"
    reader_params = count_parameters(trainer.reader)
    assert reader_params < READER_PARAM_LIMIT, f"reader has {reader_params} parameters"

    # The marked tile must repeat without a seam: shifting the tile and the
    # message map around the torus must shift the residual with them.
    # Checked on a float64 CPU copy so rounding cannot hide a real seam.
    emb = copy.deepcopy(trainer.embedder).double().cpu().eval()
    shift = TILE_PX // emb.message_grid * 16  # a whole number of message cells and of U-Net strides
    with torch.no_grad():
        tile = torch.rand(1, 3, TILE_PX, TILE_PX, dtype=torch.float64)
        bits = torch.randint(0, 2, (1, MESSAGE_BITS)).double()
        plain = emb(tile, bits)
        original_map = emb.message_map
        emb.message_map = lambda b: torch.roll(original_map(b), (shift, shift), (2, 3))
        shifted = emb(torch.roll(tile, (shift, shift), (2, 3)), bits)
    gap = (torch.roll(plain, (shift, shift), (2, 3)) - shifted).abs().max().item()
    assert gap < 1e-9, f"embedder is not seamless on the torus (max difference {gap:.2e})"

    # Flattening an axis-aligned tile-sized square at whole-pixel corners is an exact crop.
    frame = torch.rand(1, 3, 3 * TILE_PX, 3 * TILE_PX, dtype=torch.float64)
    corners = square_corners(float(TILE_PX), 1, dtype=torch.float64) + torch.tensor([200.0, 311.0])
    patch = flatten_from_corners(frame, corners, TILE_PX)
    crop = frame[:, :, 311 : 311 + TILE_PX, 200 : 200 + TILE_PX]
    gap = (patch - crop).abs().max().item()
    assert gap < 1e-9, f"flatten does not reproduce an exact crop (max difference {gap:.2e})"

    check_frame_shapes(trainer.cfg)

    # Direct mode samples the tile as an endless pattern: the identity must
    # give the tile repeated, with nothing lost at the seams.
    tile = torch.rand(1, 3, TILE_PX, TILE_PX, dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    gap = (warp_periodic(tile, identity, (2 * TILE_PX, 3 * TILE_PX)) - tile.repeat(1, 1, 2, 3)).abs().max().item()
    assert gap < 1e-9, f"periodic sampling does not reproduce the repeated tile (max difference {gap:.2e})"


def check_frame_shapes(cfg: dict) -> None:
    """The simulation and the flatten step must not care which side of the frame is longer.

    For a portrait, a landscape and a square frame: tilt a smooth surface into
    the frame, flatten a patch back out with the true corners, and compare it
    with the same patch cut straight from the surface.
    """
    geo = copy.deepcopy(cfg)
    geo["simulation"]["enabled"] = True
    geo["simulation"]["effects"] = {name: name == "perspective" for name in geo["simulation"]["effects"]}
    generator = torch.Generator().manual_seed(0)
    coarse = torch.rand(2, 3, 12, 12, generator=generator, dtype=torch.float64)
    surface = F.interpolate(coarse, size=(3 * TILE_PX, 3 * TILE_PX), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    for width, height in ((540, 960), (960, 540), (768, 768)):
        geo["simulation"]["frame_px"] = [width, height]
        sim = simulate(surface, geo, 1.0)
        assert sim.frame.shape[-2:] == (height, width), f"frame is {tuple(sim.frame.shape[-2:])}, wanted {(height, width)}"
        patch, corners = flatten_random_patch(sim.frame, sim.homography, TILE_PX, corner_jitter_px=0.0)
        assert corners[..., 0].min() >= 0 and corners[..., 0].max() <= width, "patch corners leave the frame in x"
        assert corners[..., 1].min() >= 0 and corners[..., 1].max() <= height, "patch corners leave the frame in y"
        on_surface = apply_homography(torch.linalg.inv(sim.homography), corners)
        straight = flatten_from_corners(surface, on_surface, TILE_PX)
        gap = (patch - straight).abs().max().item()
        assert gap < 0.02, f"{width}x{height} frame: flattened patch is off by {gap:.3f}"


def check_spread(trainer: Trainer) -> None:
    """The two spread options (spread.py): they measure what they claim, and gradients pass.

    Runs on copies of the models with both options switched on, whatever the
    config says, and takes no optimisation step.
    """
    cfg = copy.deepcopy(trainer.cfg)
    cfg["spread"] = {"peak_penalty": {**peak_settings(cfg), "enabled": True}, "hide": {**hide_settings(cfg), "enabled": True}}
    peak = cfg["spread"]["peak_penalty"]
    window, stride, top = int(peak["window_px"]), int(peak["stride_px"]), float(peak["top_fraction"])

    # An even mark has concentration 1. The same energy packed around the
    # tile corner, which is one spot on the torus, scores far higher, and the
    # windows must see it as one patch.
    even = torch.full((1, 3, TILE_PX, TILE_PX), 0.05, dtype=torch.float64)
    corner = torch.zeros_like(even)
    half = window // 2
    for rows in (slice(0, half), slice(TILE_PX - half, TILE_PX)):
        for cols in (slice(0, half), slice(TILE_PX - half, TILE_PX)):
            corner[:, :, rows, cols] = 0.05 * TILE_PX / window
    assert abs(concentration(even, cfg) - 1.0) < 1e-9, "an even mark must have concentration 1"
    assert abs(corner.pow(2).mean().item() - even.pow(2).mean().item()) < 1e-12
    whole = local_energy(corner, window, stride).max().item() / corner.pow(2).mean().item()
    assert abs(whole - (TILE_PX / window) ** 2) < 1e-6, "a window must cover a corner patch whole, across the tile edges"
    assert concentration(corner, cfg) > 20.0, "a corner patch must score as concentrated"
    assert peak_penalty(corner, cfg) > 20.0 * peak_penalty(even, cfg), "the penalty must cost a patch more"

    # The penalty pulls the strongest region down and leaves the rest alone.
    probe = corner.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(peak_penalty(probe, cfg), probe)
    assert grad[:, :, :half, :half].abs().min() > 0, "no gradient on the patch"
    assert grad[:, :, TILE_PX // 2, TILE_PX // 2].abs().max() == 0, "gradient away from the patch"

    # Hiding: never more than the limit, some samples untouched, kept pixels unchanged.
    hide = cfg["spread"]["hide"]
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(1)
    mask = hide_mask(256, 96, 96, float(hide["max_fraction"]), list(hide["cells"]))
    blanked = 1.0 - mask.mean(dim=(1, 2, 3))
    assert blanked.max().item() <= float(hide["max_fraction"]) + 1e-6, f"blanked {blanked.max().item():.3f} of a patch"
    assert blanked.max().item() > 0.4 * float(hide["max_fraction"]) and blanked.min().item() == 0.0
    view = torch.rand(4, 3, TILE_PX, TILE_PX)
    hidden, share = hide_regions(view, cfg)
    changed = (hidden != view).any(dim=1, keepdim=True)
    assert 0.0 < share <= float(hide["max_fraction"]) + 1e-6
    assert changed.float().mean().item() <= share + 1e-6, "pixels outside the blanked regions changed"

    # Gradients with both on: the bit loss must still reach the embedder
    # through what is left of the patch, and the penalty must reach it too.
    probe_trainer = copy.copy(trainer)
    probe_trainer.cfg = cfg
    probe_trainer.embedder = copy.deepcopy(trainer.embedder).train()
    probe_trainer.reader = copy.deepcopy(trainer.reader).train()
    tiles = torch.rand(2, 3, TILE_PX, TILE_PX, device=trainer.device)
    bits = probe_trainer.random_bits(2)
    out = probe_trainer.forward(tiles, bits, hide=True)
    assert out["hidden_fraction"] > 0.0, "nothing was blanked in the probe batch"
    bit_loss = F.binary_cross_entropy_with_logits(out["logits"], bits)
    embedder_params = [p for p in probe_trainer.embedder.parameters() if p.requires_grad]
    reader_params = [p for p in probe_trainer.reader.parameters() if p.requires_grad]
    from_bits = torch.autograd.grad(bit_loss, embedder_params + reader_params, retain_graph=True, allow_unused=True)
    from_peak = torch.autograd.grad(peak_penalty(out["residual"], cfg), embedder_params, allow_unused=True)

    def norm(grads) -> float:
        return sum(g.pow(2).sum().item() for g in grads if g is not None) ** 0.5

    to_embedder, to_reader = norm(from_bits[: len(embedder_params)]), norm(from_bits[len(embedder_params) :])
    assert all(torch.isfinite(g).all() for g in (*from_bits, *from_peak) if g is not None), "a gradient is not finite"
    assert to_embedder > 0, "with hiding on, the bit loss gives the embedder no gradient"
    assert to_reader > 0, "with hiding on, the bit loss gives the reader no gradient"
    assert norm(from_peak) > 0, "the peak penalty gives the embedder no gradient"
    torch.random.set_rng_state(generator_state)

    # The visibility columns of the spread report, on synthetic embedders whose mark is known.
    class FixedMark(nn.Module):
        def __init__(self, mark: torch.Tensor):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.register_buffer("mark", mark)

        def forward(self, tile, bits):
            return self.mark.expand_as(tile).to(tile)

    axis = torch.arange(TILE_PX, dtype=torch.float32) / TILE_PX
    wave = torch.cos(2.0 * math.pi * 18.0 * axis).view(1, 1, 1, TILE_PX).expand(1, 1, TILE_PX, TILE_PX)
    luma_mark = 0.02 * wave.expand(1, 3, TILE_PX, TILE_PX)
    gen = torch.Generator().manual_seed(3)
    # Grey noise (the same in all channels): an equal change in all three is then luminance only.
    luma_tiles = (0.2 + 0.6 * torch.rand(4, 1, TILE_PX, TILE_PX, generator=gen)).expand(4, 3, TILE_PX, TILE_PX).contiguous()
    # Blue-yellow: on flat grey, move along the blue-yellow opponent axis only (luminance and red-green stay put).
    grey = torch.full((4, 3, TILE_PX, TILE_PX), 0.5)
    axis_o3 = torch.tensor([0.0, 0.0, 0.02]).view(1, 3, 1, 1) * wave
    blue_yellow_mark = from_opponent(opponent(grey[:1]) + axis_o3) - grey[:1]
    probe_bits = torch.zeros(4, MESSAGE_BITS)
    vis_cfg = copy.deepcopy(cfg)
    vis_cfg["loss"]["visibility"] = {**visibility_settings(cfg), "enabled": True}
    off_cfg = copy.deepcopy(cfg)
    off_cfg["loss"]["visibility"] = {**visibility_settings(cfg), "enabled": False}
    measured = {}
    for name, mark, probe_tiles in (("luminance", luma_mark, luma_tiles), ("blue-yellow", blue_yellow_mark, grey)):
        embedder = FixedMark(mark)
        report = spread_report(embedder, probe_tiles, probe_bits, vis_cfg, chunk=3)
        measured[name] = report
        marked = probe_tiles + mark
        direct = tile_visibility(probe_tiles, marked, vis_cfg)
        assert abs(report["visibility_mean"] - direct["mean"].mean().item()) < 1e-4 * max(1.0, direct["mean"].mean().item()), name
        want_p90 = batch_statistic(direct["p95"], 0.9).item()
        assert abs(report["visibility_p90"] - want_p90) < 1e-4 * max(1.0, want_p90), f"{name}: p90 {report['visibility_p90']} vs {want_p90}"
        assert embedder.training, "spread_report must leave the embedder in training mode"
        quiet = spread_report(embedder, probe_tiles, probe_bits, off_cfg, chunk=3)
        assert all(math.isnan(quiet[k]) for k in ("visibility_p90", "visibility_mean", "mark_cycles", "colour_share")), "off: not NaN"
    luma, colour = measured["luminance"], measured["blue-yellow"]
    assert abs(luma["mark_cycles"] - 18.0) < 1.0, f"mark_cycles {luma['mark_cycles']:.2f} for an 18 cycle cosine"
    assert luma["colour_share"] < 0.1, f"a luminance mark has colour share {luma['colour_share']:.3f}"
    assert colour["colour_share"] > 0.8, f"a blue-yellow mark has colour share {colour['colour_share']:.3f}"
    assert luma["visibility_p90"] > 0.0 and colour["visibility_p90"] > 0.0
    print(
        f"spread checks passed: bit-loss gradient norm {to_embedder:.3g} at the embedder and {to_reader:.3g} at the "
        f"reader with {out['hidden_fraction'] * 100:.0f} percent of the patch blanked; "
        f"peak-penalty gradient norm {norm(from_peak):.3g} at the embedder; "
        f"18 cycle luminance mark: {luma['mark_cycles']:.2f} cycles, colour share {luma['colour_share']:.3f}; "
        f"blue-yellow mark: colour share {colour['colour_share']:.3f}"
    )


def check_texture_mask(trainer: Trainer) -> None:
    """loss.texture_mask: flat colour costs flat_weight, detail costs near busy_weight, and the loss uses it.

    Runs on a fresh trainer with the mask and the peak penalty on and the
    image loss at full weight, whatever the config says.
    """
    cfg = copy.deepcopy(trainer.cfg)
    cfg["loss"]["texture_mask"] = {**texture_settings(cfg), "enabled": True}
    cfg["loss"]["lpips_weight"] = 0.0
    cfg["spread"] = {**(cfg.get("spread") or {}), "peak_penalty": {**peak_settings(cfg), "enabled": True}}
    s = texture_settings(cfg)
    flat_w, busy_w = float(s["flat_weight"]), float(s["busy_weight"])

    # Left half flat grey, right half a sharp 8 px checkerboard.
    tile = torch.full((1, 3, TILE_PX, TILE_PX), 0.5)
    y, x = torch.meshgrid(torch.arange(TILE_PX), torch.arange(TILE_PX), indexing="ij")
    checks = ((y // 8 + x // 8) % 2).float() * 0.6 + 0.2
    tile[:, :, :, TILE_PX // 2 :] = checks[:, TILE_PX // 2 :]
    cost = texture_weights(tile, cfg)
    assert cost.shape == (1, 1, TILE_PX, TILE_PX)
    assert cost.min().item() >= busy_w - 1e-6 and cost.max().item() <= flat_w + 1e-6, "weight outside [busy, flat]"
    margin = int(s["window_px"]) // 2 + 5
    flat_part = cost[..., :, : TILE_PX // 2 - margin]
    busy_part = cost[..., :, TILE_PX // 2 + margin :]
    assert (flat_part - flat_w).abs().max().item() < 1e-6, "flat colour must cost flat_weight"
    assert busy_part.max().item() < busy_w + 0.05 * (flat_w - busy_w), "detail must cost close to busy_weight"
    assert (texture_weights(torch.full_like(tile, 0.3), cfg) - flat_w).abs().max().item() < 1e-6

    # The training step uses the weighted L2 and the weighted peak penalty.
    probe = Trainer(cfg, trainer.device)
    probe.embedder.load_state_dict(trainer.embedder.state_dict())
    probe.reader.load_state_dict(trainer.reader.state_dict())
    probe.cfg["schedule"]["image_loss"]["follow_accuracy"] = None
    probe.image_ramp_start = -(10**9)  # the plain ramp, long finished: weight 1
    assert probe.image_loss_scale() == 1.0
    tiles = torch.rand(2, 3, TILE_PX, TILE_PX)
    tiles[0] = 0.4  # one flat tile, one busy one
    bits = probe.random_bits(2)
    probe.random_bits = lambda batch: bits  # the same message in the step as in the expectation below
    probe.embedder.train()
    with torch.no_grad():
        residual = probe.embedder(tiles.to(probe.device), bits)
        want_l2 = (texture_weights(tiles.to(probe.device), cfg) * residual.pow(2)).mean().item()
        want_peak = peak_penalty(residual * texture_weights(tiles.to(probe.device), cfg).sqrt(), cfg).item()
    m = probe.train_step(tiles)
    assert "l2_texture_loss" in m and "peak_loss" in m, "the texture-weighted terms were not computed"
    assert abs(m["l2_texture_loss"] - want_l2) < 1e-4 * want_l2, f"L2 in the step {m['l2_texture_loss']:.6g}, texture-weighted {want_l2:.6g}"
    assert abs(m["peak_loss"] - want_peak) < 1e-4 * want_peak, f"peak in the step {m['peak_loss']:.6g}, texture-weighted {want_peak:.6g}"
    expected = (
        cfg["loss"]["bit_weight"] * m["bit_loss"]
        + cfg["loss"]["l2_weight"] * m["l2_texture_loss"]
        + float(cfg["spread"]["peak_penalty"]["weight"]) * m["peak_loss"]
    )
    assert abs(m["loss"] - expected) < 1e-4 * max(1.0, abs(expected)), f"loss {m['loss']:.6f}, terms give {expected:.6f}"

    # Gradient of the weighted terms reaches the embedder and is finite.
    probe.embedder.train()
    residual = probe.embedder(tiles.to(probe.device), probe.random_bits(2))
    weights = texture_weights(tiles.to(probe.device), cfg)
    term = (weights * residual.pow(2)).mean() + peak_penalty(residual * weights.sqrt(), cfg)
    grads = torch.autograd.grad(term, [p for p in probe.embedder.parameters() if p.requires_grad], allow_unused=True)
    norm = sum(g.pow(2).sum().item() for g in grads if g is not None) ** 0.5
    assert norm > 0 and all(torch.isfinite(g).all() for g in grads if g is not None), "no finite gradient from the mask"
    print(
        f"texture mask checks passed: flat colour costs {flat_w:g}, the checkerboard at most "
        f"{busy_part.max().item():.3f} (floor {busy_w:g}); the loss uses the weighted L2 and peak penalty; "
        f"gradient norm {norm:.3g} at the embedder"
    )


def check_visibility() -> None:
    """visibility.py: S-CIELAB with masking ranks what the eye sees on a 50 mm tile at 50 cm.

    Needs no trainer. Runs on the CPU, and again on CUDA when there is a GPU.
    """
    import labelgen
    from data import label_tile
    from visibility import (
        batch_statistic,
        delta_e_map,
        filter_spectra,
        from_opponent,
        masking_factor,
        opponent,
        pixels_per_degree,
        tile_visibility,
        visibility_settings,
    )

    cfg = load_config(HERE / "configs" / "default.yaml")
    s = visibility_settings(cfg)
    assert s == {"enabled": False, "distance_mm": 500, "masking_floor": None}, f"default visibility settings {s}"
    for bad in ({"distance_mm": 0}, {"masking_floor": 0.0}, {"masking_floor": 1.5}):
        try:
            visibility_settings({"loss": {"visibility": bad}})
        except ValueError:
            continue
        raise AssertionError(f"loss.visibility {bad} was accepted")
    ppd = pixels_per_degree(50, 500)
    assert abs(ppd - 67.1) < 0.1, f"pixels per degree {ppd:.3f}, want 67.1"
    spectra = filter_spectra(TILE_PX, TILE_PX, ppd, torch.device("cpu"), torch.float32)
    assert spectra.shape == (3, TILE_PX, TILE_PX)
    assert (spectra[:, 0, 0] - 1.0).abs().max().item() < 1e-4, f"filter DC gains {spectra[:, 0, 0].tolist()}, want 1"

    grey = torch.full((1, 3, TILE_PX, TILE_PX), 0.5)
    x = torch.arange(TILE_PX, dtype=torch.float64)

    def cosine(cycles: int) -> torch.Tensor:
        """Vertical stripes, (1,1,H,W): a cosine with a whole number of cycles across the tile, so it repeats seamlessly."""
        return torch.cos(2 * torch.pi * cycles * x / TILE_PX).float().expand(TILE_PX, TILE_PX)[None, None]

    def mean_delta_e(clean: torch.Tensor, marked: torch.Tensor) -> float:
        return delta_e_map(clean, marked, ppd).mean().item()

    base = opponent(grey)
    assert (from_opponent(base) - grey).abs().max().item() < 1e-5, "from_opponent does not undo opponent"

    def opponent_swing(channel: int, cycles: int) -> float:
        """Mean delta E of a cosine of amplitude 0.01 in one opponent channel of mid-grey."""
        shifted = base.clone()
        shifted[:, channel : channel + 1] += 0.01 * cosine(cycles)
        return mean_delta_e(grey, from_opponent(shifted))

    # The eye sees a coarse luminance pattern far better than a fine one at the same amplitude. In
    # S-CIELAB's luminance channel alone the drop from 18 to 74 cycles per tile is 3.5x. A grey grating
    # (equal RGB) also moves the red-green channel a little, since grey is not on that channel's zero,
    # and that channel is filtered differently: the faithful model then gives 2.3x, hence 2 there.
    coarse = mean_delta_e(grey, grey + 0.02 * cosine(18))
    fine = mean_delta_e(grey, grey + 0.02 * cosine(74))
    assert coarse >= 2.0 * fine, f"18 cycles per tile scores {coarse:.4f}, 74 cycles {fine:.4f}: ratio {coarse / fine:.2f}, want >= 2"
    luminance_coarse, luminance_fine = opponent_swing(0, 18), opponent_swing(0, 74)
    assert luminance_coarse >= 3.0 * luminance_fine, (
        f"luminance channel: 18 cycles {luminance_coarse:.4f}, 74 cycles {luminance_fine:.4f}, want a ratio >= 3"
    )

    # At the same spacing and the same opponent-space amplitude, blue-yellow is less visible than luminance.
    swing = {"luminance": opponent_swing(0, 23), "blue-yellow": opponent_swing(2, 23)}
    assert swing["blue-yellow"] < swing["luminance"], f"blue-yellow {swing['blue-yellow']:.4f} not below luminance {swing['luminance']:.4f}"

    # Masking: 1 on flat colour, never below the floor, and the same residual costs less on a busy label.
    flat_colour = torch.tensor([0.85, 0.32, 0.18])
    flat = flat_colour.view(1, 3, 1, 1).expand(1, 3, TILE_PX, TILE_PX).clone()
    assert (masking_factor(flat, cfg) - 1.0).abs().max().item() < 1e-6, "masking must be 1 on flat colour"
    labels = torch.stack([label_tile(seed) for seed in labelgen.ACCEPTANCE_SEEDS[:8]])
    masks = masking_factor(labels, cfg)
    assert masks.min().item() >= 0.408 and masks.max().item() <= 1.0 + 1e-6, f"masking in [{masks.min().item():.4f}, {masks.max().item():.4f}]"
    # Acceptance seed 4: dark text over most of a pale grey label. Its own average colour, flat, is the
    # comparison; on that near-grey label the unmasked delta E hardly depends on which colours the residual
    # lands on, so the difference below is the masking's. (On a red label with white text, the colour
    # alone changes delta E twofold.)
    busy = labels[4:5]
    paper = busy.mean(dim=(2, 3), keepdim=True).expand_as(busy).clone()
    residual = 0.02 * cosine(23).expand(1, 3, TILE_PX, TILE_PX) * torch.tensor([1.0, -0.5, 0.3]).view(1, 3, 1, 1)
    raw_flat, raw_busy = mean_delta_e(paper, paper + residual), mean_delta_e(busy, busy + residual)
    assert abs(raw_busy / raw_flat - 1.0) < 0.15, f"unmasked delta E {raw_busy:.4f} on the label, {raw_flat:.4f} flat: not a fair comparison"
    on_flat = tile_visibility(paper, paper + residual, cfg)["mean"].item()
    on_busy = tile_visibility(busy, busy + residual, cfg)["mean"].item()
    assert abs(on_flat - raw_flat) < 1e-6 * raw_flat, "masking changed the score on flat colour"
    assert on_flat > 1.3 * on_busy, f"the same residual scores {on_flat:.4f} on flat colour and {on_busy:.4f} on a busy label"

    # Nothing marked: zero, with a finite gradient, also on pure white and pure black.
    for name, clean in (("grey", grey), ("white", torch.ones_like(grey)), ("black", torch.zeros_like(grey)), ("label", busy)):
        marked = clean.clone().requires_grad_(True)
        out = tile_visibility(clean, marked, cfg)
        assert out["map"].shape == (1, 1, TILE_PX, TILE_PX) and out["mean"].shape == (1,) and out["p95"].shape == (1,)
        assert out["map"].abs().max().item() < 1e-4, f"{name}: marked == clean scores {out['map'].abs().max().item():.3g}"
        (out["mean"].sum() + out["p95"].sum()).backward()
        assert torch.isfinite(marked.grad).all(), f"{name}: gradient not finite with nothing marked"

    def gradient_check(device: torch.device) -> tuple[float, float, float]:
        """Mean, p95 and gradient norm for a residual on the busy label, on one device."""
        residual_d = (0.03 * torch.randn(1, 3, TILE_PX, TILE_PX, generator=torch.Generator().manual_seed(0))).to(device).requires_grad_(True)
        clean_d = busy.to(device)
        out = tile_visibility(clean_d, clean_d + residual_d, cfg)
        out["mean"].sum().backward()
        grad = residual_d.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max().item() > 0, f"{device}: no finite, nonzero gradient"
        return out["mean"].item(), out["p95"].item(), grad.norm().item()

    cpu_mean, cpu_p95, cpu_grad = gradient_check(torch.device("cpu"))
    if torch.cuda.is_available():
        gpu_mean, gpu_p95, gpu_grad = gradient_check(torch.device("cuda"))
        assert abs(gpu_mean - cpu_mean) < 1e-3 * cpu_mean and abs(gpu_p95 - cpu_p95) < 1e-3 * cpu_p95, "CUDA and CPU disagree"
        assert abs(gpu_grad - cpu_grad) < 1e-3 * cpu_grad, "CUDA and CPU gradients disagree"

    assert abs(batch_statistic(torch.arange(11.0)).item() - 9.0) < 1e-5, "batch_statistic is not the 90th percentile over tiles"

    # Circular filtering: rolling the tile rolls the map and changes nothing else. A flat clean tile, so the
    # masking (padded by reflection, not circularly) stays constant and only the filtering is tested.
    noise = 0.03 * torch.randn(1, 3, TILE_PX, TILE_PX, generator=torch.Generator().manual_seed(1))
    straight = tile_visibility(flat, flat + noise, cfg)
    rolled = tile_visibility(flat, flat + noise.roll((100, 100), dims=(2, 3)), cfg)
    moved = (rolled["map"] - straight["map"].roll((100, 100), dims=(2, 3))).abs().max().item()
    assert moved < 1e-4 * straight["map"].max().item(), f"rolling the tile changed the map by up to {moved:.3g}"  # float32 rounding only
    assert abs(rolled["mean"].item() - straight["mean"].item()) < 1e-5 * straight["mean"].item() + 1e-6
    assert abs(rolled["p95"].item() - straight["p95"].item()) < 1e-4 * straight["p95"].item() + 1e-6
    print(
        f"visibility checks passed: {ppd:.2f} px per degree; a grey grating at 18 cycles per tile {coarse / fine:.2f}x "
        f"as visible as at 74 ({luminance_coarse / luminance_fine:.2f}x in the luminance channel); blue-yellow {swing['blue-yellow'] / swing['luminance']:.2f}x luminance at 23; "
        f"the same residual {on_flat / on_busy:.2f}x as visible on flat colour as on a busy label "
        f"(masking down to {masks.min().item():.3f}); zero and finite gradients with nothing marked; "
        f"the same when rolled 100 px"
    )


def check_calibration_levels() -> None:
    """calibration_sheet.py: scaled tiles hit their levels; --seen applies the design's rules."""
    from calibration_sheet import LEVELS, decide, scale_to_level
    from data import label_tile
    import labelgen

    cfg = load_config(HERE / "configs" / "default.yaml")
    x = torch.arange(TILE_PX, dtype=torch.float32)
    wave = torch.cos(2 * torch.pi * 12 * x / TILE_PX).expand(TILE_PX, TILE_PX)
    residual = (wave[None] * torch.tensor([0.05, 0.03, -0.04]).view(3, 1, 1)).unsqueeze(0)
    flat = torch.full((1, 3, TILE_PX, TILE_PX), 0.5)
    busy = label_tile(labelgen.ACCEPTANCE_SEEDS[0])[None]
    from calibration_sheet import quantise

    for name, clean in (("flat", quantise(flat)), ("busy", quantise(busy))):
        for level in LEVELS:
            marked = scale_to_level(clean, residual, level, cfg)
            got = float(tile_visibility(clean, marked, cfg)["p95"][0])
            assert abs(got - level) <= 0.05 * level, f"{name} tile at level {level}: p95 {got:.3f}"
    try:
        scale_to_level(flat, residual * 1e-4, 8.0, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("an unreachable level was not refused")

    # The tile as saved: 8-bit PNGs read back give the level too, and _recompute reads them.
    from calibration_sheet import _p95, _read, _recompute, quantise

    with tempfile.TemporaryDirectory(prefix="tilemark_calibration_") as tmp:
        for name, clean in (("flat", flat), ("busy", busy)):
            for level in (0.5, 1.0):
                marked = scale_to_level(clean, residual, level, cfg)
                save_image(quantise(clean)[0], Path(tmp) / "pairs" / f"01_left.png")
                save_image(marked[0], Path(tmp) / "pairs" / f"01_right.png")
                back = _p95(_read(Path(tmp) / "pairs" / "01_left.png"), _read(Path(tmp) / "pairs" / "01_right.png"), cfg)
                assert abs(back - level) <= 0.05 * level, f"{name} tile at level {level}, read back from PNG: p95 {back:.3f}"
        key = {"directory": tmp, "config": str(HERE / "configs" / "default.yaml")}
        pair = {"pair": 1, "marked": "right"}
        for floor in (0.5, 1.0):
            floored = {**cfg, "loss": {**cfg["loss"], "visibility": {**(cfg["loss"].get("visibility") or {}), "masking_floor": floor}}}
            want = _p95(_read(Path(tmp) / "pairs" / "01_left.png"), _read(Path(tmp) / "pairs" / "01_right.png"), floored)
            got = _recompute(key, pair, floor)
            assert abs(got - want) < 1e-5 * max(1.0, want), f"_recompute at floor {floor}: {got} against {want}"

    # --seen on a synthetic key: per level one pair each of two flat labels, a busy label, a 74-cycle
    # luminance grating and a 23-cycle blue-yellow grating.
    pairs = []
    for i, level in enumerate(LEVELS):
        for j, (kind, extra) in enumerate(
            [("flat label", {}), ("flat label", {}), ("busy label", {}),
             ("grating", {"channel": "luminance", "cycles": 74}), ("grating", {"channel": "blue-yellow", "cycles": 23})]
        ):
            pairs.append({"pair": 1 + 5 * i + j, "kind": kind, "level": level, **extra})
    key = {"levels": list(LEVELS), "pairs": pairs}

    def at(kind: str, level: float, k: int = 0) -> int:
        return [p["pair"] for p in pairs if p["kind"] == kind and p["level"] == level][k]

    p95 = lambda p, floor: 10.0 * floor
    d = decide(key, [], p95)
    assert d["target"] == 8 and d["floor"] is None and not d["wait"] and not d["fired"], f"nothing seen: {d}"
    d = decide(key, [at("flat label", 8)], p95)
    assert d["target"] == 5 and not d["wait"], f"flat seen at 8: {d}"
    d = decide(key, [at("flat label", 3, 1), at("flat label", 8)], p95)
    assert d["target"] == 2, f"flat seen at 3 and 8: the target stays under the lowest seen level: {d}"
    d = decide(key, [at("flat label", 0.5)], p95)
    assert d["wait"] and d["target"] is None, f"flat seen at the lowest level: {d}"
    d = decide(key, [at("flat label", level, k) for level in LEVELS for k in (0, 1)], p95)
    assert d["wait"] and d["target"] is None and "every level seen: the run waits" in d["fired"][0], f"every level seen: {d}"
    d = decide(key, [at("flat label", 8), at("busy label", 3)], p95)
    assert d["target"] == 5 and d["floor"] == 0.6, f"busy seen at 3, target 5: {d}"
    d = decide(key, [at("flat label", 8), at("busy label", 5)], p95)
    assert d["floor"] == 0.6, f"busy seen at the target: {d}"
    d = decide(key, [at("flat label", 5), at("busy label", 8)], p95)
    assert d["target"] == 3 and d["floor"] is None, f"busy seen above the target must not move the floor: {d}"
    d = decide(key, [at("busy label", 1)], lambda p, floor: 1.0)
    assert d["floor"] == 1.0 and any("no floor" in line for line in d["fired"]), f"floor cannot reach the target: {d}"
    d = decide(key, [at("flat label", 8), at("grating", 3, 0)], p95)
    assert d["target"] == 2, f"74-cycle grating seen at 3: {d}"
    d = decide(key, [at("grating", 5, 0), at("grating", 3, 0)], p95)
    assert d["target"] == 2, f"74-cycle gratings seen at 3 and 5: {d}"
    d = decide(key, [at("grating", 8, 0)], p95)
    assert d["target"] == 5, f"74-cycle grating seen at 8: {d}"
    d = decide(key, [at("flat label", 3), at("grating", 5, 0)], p95)
    assert d["target"] == 2, f"flat seen at 3 puts the target at 2; the grating at 5 is above it: {d}"
    d = decide(key, [at("grating", 8, 1)], p95)
    assert d["target"] == 8, f"a 23-cycle grating does not lower the target: {d}"
    d = decide(key, [at("grating", 0.5, 0)], p95)
    assert d["wait"], f"the 74-cycle grating seen at the lowest level: {d}"
    print("calibration checks passed: scaled tiles within 5 percent of 0.5 to 8, an unreachable level refused, --seen rules")


def check_follow_accuracy(trainer: Trainer) -> None:
    """schedule.image_loss.follow_accuracy: the image-loss weight rises while the mark reads and falls when it does not.

    Drives the schedule alone, with made-up bit accuracies, on copies that
    share the models; no optimisation step is taken.
    """
    base = copy.deepcopy(trainer.cfg)
    base["schedule"]["image_loss"].update(acc_threshold=0.95, acc_ema=0.0, ramp_steps=10, follow_accuracy=None)
    following = copy.deepcopy(base)
    following["schedule"]["image_loss"]["follow_accuracy"] = {"floor": 0.95, "band": 0.01}

    def drive(cfg: dict, accuracies: list[float]) -> tuple[list[float], Trainer]:
        """The scale each step would train with. acc_ema 0 makes the smoothed accuracy the step's own."""
        probe = copy.copy(trainer)
        probe.cfg, probe.step, probe.acc_ema, probe.image_ramp_start, probe.follow_scale = cfg, 0, 0.5, None, 0.0
        scales = []
        for accuracy in accuracies:
            scales.append(probe.image_loss_scale())
            probe.step += 1
            probe.update_schedule(accuracy)
        return scales, probe

    def close(a: list[float], b: list[float]) -> bool:
        return len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b))

    # Off until the trigger, whatever the accuracy does below it.
    scales, _ = drive(following, [0.90, 0.94, 0.90])
    assert scales == [0.0, 0.0, 0.0], f"the image loss came on before the trigger: {scales}"
    # Up while at or above the floor, held inside the band, down below it, up again; never outside 0 to 1.
    story = [0.96] + [0.96] * 4 + [0.945] * 3 + [0.93] * 2 + [0.95] * 3 + [0.5] * 20 + [0.99] * 15
    scales, probe = drive(following, story)
    wanted = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1] + [0.0] * 16
    wanted += [0.1 * n for n in range(1, 11)] + [1.0] * 4
    assert close(scales, wanted), f"follow_accuracy moved the scale wrongly:\n{[round(s, 3) for s in scales]}"
    assert max(scales) <= 1.0 and min(scales) >= 0.0 and abs(probe.image_loss_scale() - 1.0) < 1e-9
    # While accuracy never drops it is the plain ramp, step for step.
    steady = [0.96] * 16
    assert close(drive(following, steady)[0], drive(base, steady)[0]), "with accuracy above the floor it must be the plain ramp"
    # Without the setting, a drop in accuracy changes nothing (the behaviour before this option).
    assert close(drive(base, story)[0], [0.0] + [min(1.0, n / 10) for n in range(len(story) - 1)]), "the plain ramp changed"

    # The scale is in the checkpoint and comes back from it.
    scales, probe = drive(following, [0.96] * 5 + [0.5] * 2)
    assert abs(probe.image_loss_scale() - 0.2) < 1e-9
    with tempfile.TemporaryDirectory(prefix="tilemark_follow_") as tmp:
        path = save_checkpoint(probe, tmp, keep=0)
        state = torch.load(path, map_location=trainer.device, weights_only=True)
        assert abs(state["image_scale"] - 0.2) < 1e-9, "the scale is not in the checkpoint"
        other = copy.copy(probe)
        other.follow_scale, other.step = 0.9, 0
        load_checkpoint(other, path)
        assert other.step == probe.step and abs(other.image_loss_scale() - 0.2) < 1e-9, "the scale did not come back"
    print(
        "schedule checks passed: with follow_accuracy the image-loss weight rises at or above the floor, holds in "
        "the band, falls below it, stays within 0 to 1, and is saved and restored; without it the ramp is unchanged"
    )


def check_follow_visibility(trainer: Trainer) -> None:
    """schedule.image_loss.follow_visibility: the visibility weight rises while the mark is too visible and falls below the target.

    Drives update_visibility alone, with made-up values, on copies that share the models.
    """
    base = copy.deepcopy(trainer.cfg)
    base["schedule"]["image_loss"]["follow_accuracy"] = None
    base["schedule"]["image_loss"]["follow_visibility"] = {"target": 2.0, "ema": 0.0, "rate": 0.1, "init_weight": 0.01}

    def make(cfg: dict) -> Trainer:
        probe = copy.copy(trainer)
        probe.cfg, probe.step = cfg, 0
        probe.image_ramp_start = None
        probe.vis_weight, probe.vis_ema = cfg["schedule"]["image_loss"]["follow_visibility"].get("init_weight", 0.01), None
        return probe

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))

    settings = make(base).visibility_follow_settings()
    assert settings == {
        "target": 2.0, "quantile": 0.9, "ema": 0.0, "low": 0.9, "rate": 0.1,
        "init_weight": 0.01, "min_weight": 0.01, "max_weight": 1000.0,
    }, f"follow_visibility defaults are wrong: {settings}"

    probe = make(base)
    probe.update_visibility(3.0)
    assert probe.vis_weight == 0.01 and probe.vis_ema is None, "the weight moved before the trigger"
    probe.image_ramp_start = 0
    for n in range(1, 6):
        probe.update_visibility(3.0)
        assert near(probe.vis_weight, 0.01 * 1.1**n), f"too visible: weight {probe.vis_weight} after {n} steps"
    top = probe.vis_weight
    for n in range(1, 4):
        probe.update_visibility(1.7)
        assert near(probe.vis_weight, top / 1.1**n), f"below 0.9 x target: weight {probe.vis_weight} after {n} steps"
    held = probe.vis_weight
    for value in (1.9, 1.8, 2.0):
        probe.update_visibility(value)
        assert probe.vis_weight == held, f"the weight moved inside the band at {value}"
    for _ in range(200):
        probe.update_visibility(1.0)
    assert probe.vis_weight == 0.01, f"the weight fell below min_weight: {probe.vis_weight}"
    for _ in range(300):
        probe.update_visibility(9.0)
    assert probe.vis_weight == 1000.0, f"the weight is not clamped at max_weight: {probe.vis_weight}"

    # The first value after the trigger is the EMA (no pull toward 0); then it is smoothed.
    smoothed = copy.deepcopy(base)
    smoothed["schedule"]["image_loss"]["follow_visibility"]["ema"] = 0.98
    probe = make(smoothed)
    probe.image_ramp_start = 0
    probe.update_visibility(3.0)
    assert probe.vis_ema == 3.0, f"the first value did not set the EMA: {probe.vis_ema}"
    probe.update_visibility(1.0)
    assert near(probe.vis_ema, 0.98 * 3.0 + 0.02 * 1.0), f"the EMA is not smoothed: {probe.vis_ema}"

    # Both values are in the checkpoint and come back from it; an older checkpoint gives the start values.
    probe.vis_weight, probe.vis_ema = 0.123456789, 2.718281828
    with tempfile.TemporaryDirectory(prefix="tilemark_vis_") as tmp:
        path = save_checkpoint(probe, tmp, keep=0)
        other = make(smoothed)
        other.vis_weight, other.vis_ema = 5.0, 5.0
        load_checkpoint(other, path)
        assert other.vis_weight == 0.123456789 and other.vis_ema == 2.718281828, "vis_weight and vis_ema did not come back"
        state = torch.load(path, map_location=trainer.device, weights_only=True)
        del state["vis_weight"], state["vis_ema"]
        old = Path(tmp) / "old.pt"
        torch.save(state, old)
        load_checkpoint(other, old)
        assert other.vis_weight == 0.01 and other.vis_ema is None, "a checkpoint without them must give init_weight and None"

    # Bad settings fail at start.
    for follow_visibility, follow_accuracy in (({"target": None}, None), ({"target": 2.0}, {"floor": 0.95, "band": 0.01})):
        bad = copy.deepcopy(trainer.cfg)
        bad["schedule"]["image_loss"]["follow_visibility"] = follow_visibility
        bad["schedule"]["image_loss"]["follow_accuracy"] = follow_accuracy
        try:
            Trainer(bad, trainer.device)
        except ValueError:
            continue
        raise AssertionError(f"Trainer accepted follow_visibility {follow_visibility} with follow_accuracy {follow_accuracy}")
    print(
        "visibility schedule checks passed: the weight is untouched before the trigger, rises x(1+rate) while the "
        "value is above the target, falls below low x target, holds between, stays within its bounds, starts the "
        "EMA at the first value, and is saved and restored"
    )


def check_visibility_step(trainer: Trainer) -> None:
    """The visibility loss in a training step, its metrics, the progress columns and notify.py's reading of them."""
    import csv

    import notify
    from runlog import PROGRESS_COLUMNS, ProgressLog, RunClock

    base = copy.deepcopy(trainer.cfg)
    sched = base["schedule"]["image_loss"]
    sched.update(follow_accuracy=None, ramp_steps=10)
    sched["follow_visibility"] = {"target": 2.0, "ema": 0.0, "rate": 0.1, "init_weight": 0.01}
    base["loss"].update(l2_weight=0.0, lpips_weight=0.0)
    base["loss"]["visibility"] = {"enabled": True, "distance_mm": 500, "masking_floor": None}
    base["loss"]["texture_mask"] = {**(base["loss"].get("texture_mask") or {}), "enabled": False}
    base["spread"] = {**(base.get("spread") or {}), "peak_penalty": {"enabled": False}}
    tiles = torch.from_numpy(
        np.stack([procedural_image(np.random.default_rng(k), TILE_PX) for k in range(4)])
    ).permute(0, 3, 1, 2).float().div(255.0)

    def run(cfg: dict, weight: float | None = None, image_on: bool = True) -> tuple[dict, Trainer, list[torch.Tensor]]:
        torch.manual_seed(1234)
        probe = Trainer(cfg, trainer.device)
        if image_on:
            probe.image_ramp_start, probe.step = 0, 10**6
        if weight is not None:
            probe.vis_weight = weight
        metrics = probe.train_step(tiles)
        grads = [p.grad.detach().clone() for p in probe.embedder.parameters() if p.grad is not None]
        return metrics, probe, grads

    m1, probe, g1 = run(base, 0.01)
    for key in ("visibility", "visibility_mean", "vis_weight"):
        assert key in m1 and math.isfinite(m1[key]), f"metric {key} missing or not finite: {m1.get(key)}"
    assert m1["visibility_mean"] > 0, "the visibility of a mark is not above 0"
    assert sum(g.abs().sum().item() for g in g1) > 0, "the embedder has no gradient"
    _, _, g2 = run(base, 5.0)
    assert any(not torch.allclose(a, b) for a, b in zip(g1, g2)), "the embedder gradient does not depend on vis_weight"
    assert probe.vis_ema is not None and math.isfinite(probe.vis_ema), "update_visibility was not called"

    # Image loss off: the metrics are still there (computed without gradient), the weight stays.
    m0, probe0, _ = run(base, None, image_on=False)
    assert math.isfinite(m0["visibility"]) and probe0.vis_ema is None, "ramp 0: metric missing or the controller moved"

    # Disabled: old configs train as before, the loss is the old formula.
    old = copy.deepcopy(base)
    old["loss"]["visibility"]["enabled"] = False
    old["loss"].update(l2_weight=1.0)
    mo, _, _ = run(old, None)
    assert "visibility" not in mo and "vis_weight" not in mo, "visibility metrics with loss.visibility off"
    expected = old["loss"]["bit_weight"] * mo["bit_loss"] + 1.0 * old["loss"]["l2_weight"] * mo["l2_loss"]
    assert abs(mo["loss"] - expected) <= 1e-4 * max(1.0, abs(expected)), f"old loss changed: {mo['loss']} vs {expected}"

    # Without follow_visibility the weight is 1.0.
    plain = copy.deepcopy(base)
    plain["schedule"]["image_loss"]["follow_visibility"] = None
    mp, _, _ = run(plain, None)
    expected = plain["loss"]["bit_weight"] * mp["bit_loss"] + mp["visibility_mean"]
    assert abs(mp["loss"] - expected) <= 1e-4 * max(1.0, abs(expected)), f"weight 1.0 loss: {mp['loss']} vs {expected}"

    ev = probe.evaluate(tiles)
    assert math.isfinite(ev["visibility"]), "evaluate has no visibility"

    # progress.csv: new columns at the end, read by notify.py; an old header is set aside as before.
    assert PROGRESS_COLUMNS[-2:] == ["visibility", "vis_weight"], f"columns: {PROGRESS_COLUMNS}"
    with tempfile.TemporaryDirectory(prefix="tilemark_vis_log_") as tmp:
        clock = RunClock(tmp, None)
        log = ProgressLog(tmp)
        log.write(100, 0.9, 30.0, 100, clock, 0.5, 1.75, 0.02)
        rows = notify.read_rows(Path(tmp) / "progress.csv")
        assert rows and rows[-1]["visibility"] == "1.7500" and rows[-1]["vis_weight"].startswith("0.02"), rows
        view = {"progress": rows, "spread": []}
        spread_row = {"step": "100", "local_time": "t", "fixed_place": "1", "strongest_spot": "1", "square_share": "0.1",
                      "busy_share": "0.5", "psnr_db": "30", "psnr_fixed_db": "30", "bit_acc": "0.9"}
        for what, text in (
            ("situation", notify.situation("run", view, 1000)),
            ("report_text", notify.report_text("run", spread_row, view, 1000)[1]),
        ):
            assert "visibility 1.75" in text and "weight 0.02" in text, f"notify.{what} does not show the visibility and its weight"
        old_dir = Path(tmp) / "old"
        old_dir.mkdir()
        with open(old_dir / "progress.csv", "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(PROGRESS_COLUMNS[:-2])
            csv.writer(f).writerow(["t", "1", "0.5", "20", "1", "0.1", "", "0"])
        old_rows = notify.read_rows(old_dir / "progress.csv")
        notify.situation("run", {"progress": old_rows}, 1000)  # an older file: no error
        notify.report_text("run", spread_row, {"progress": old_rows, "spread": []}, 1000)
        ProgressLog(old_dir)
        assert len(list(old_dir.glob("progress_before_*.csv"))) == 1, "an old header was not set aside"
    print(
        "visibility step checks passed: metrics finite, the embedder gets a gradient that follows vis_weight, "
        "the old loss is unchanged with it off, weight 1.0 without the controller, and the log columns read"
    )


def check_center_fraction() -> None:
    """data.center_fraction: crops come from the middle of the image only, and 1.0 changes nothing."""
    size, low, high = 500, 100, 400  # a red frame around a green middle: the central 60 percent of each side
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    pixels[low:high, low:high] = (0, 255, 0)
    state = torch.random.get_rng_state()
    with tempfile.TemporaryDirectory(prefix="tilemark_center_") as tmp:
        Image.fromarray(pixels).save(Path(tmp) / "frame.png")
        middle = ImageFolderDataset(tmp, image_size=64, min_crop_scale=0.35, center_fraction=0.6)
        whole = ImageFolderDataset(tmp, image_size=64, min_crop_scale=0.35)
        torch.manual_seed(0)
        red_in_middle = max(middle[0][0].max().item() for _ in range(64))
        torch.manual_seed(0)
        whole_crops = [whole[0] for _ in range(64)]
        old = transforms.Compose(
            [
                transforms.RandomResizedCrop(64, scale=(0.35, 1.0), ratio=(0.8, 1.25), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        torch.manual_seed(0)
        with Image.open(Path(tmp) / "frame.png") as image:
            old_crops = [old(image.convert("RGB")) for _ in range(64)]
        for bad in (0.0, 1.5):
            try:
                ImageFolderDataset(tmp, center_fraction=bad)
            except ValueError:
                continue
            raise AssertionError(f"center_fraction {bad} was accepted")
    torch.random.set_rng_state(state)
    assert red_in_middle == 0.0, f"a crop with center_fraction 0.6 reached the outer frame (red {red_in_middle:.3f})"
    assert max(crop[0].max().item() for crop in whole_crops) > 0.0, "no crop of the whole image touched the frame"
    assert all(torch.equal(a, b) for a, b in zip(whole_crops, old_crops)), "center_fraction 1.0 changed the crops"
    print("crop checks passed: center_fraction 0.6 keeps every crop inside the middle; 1.0 gives the crops it always gave")


def check_holdout() -> None:
    """data.holdout: the images kept back never reach training, and 0 keeps none back."""
    with tempfile.TemporaryDirectory(prefix="tilemark_holdout_") as tmp:
        make_tiny_dataset(Path(tmp), count=12, size=32)
        everything = ImageFolderDataset(tmp, image_size=16)
        train = ImageFolderDataset(tmp, image_size=16, holdout=5)
        again = ImageFolderDataset(tmp, image_size=16, holdout=5)
        held = holdout_images(tmp, 5)
        try:
            ImageFolderDataset(tmp, image_size=16, holdout=12)
        except ValueError:
            pass
        else:
            raise AssertionError("a holdout as large as the dataset was accepted")
    assert len(everything) == 12 and len(train) == 7 and len(held) == 5, "the split does not add up"
    assert not set(train.files) & set(held), "an image is both trained on and held out"
    assert set(train.files) | set(held) == set(everything.files), "the split lost an image"
    assert train.files == again.files, "the split is not the same every time"
    print("holdout checks passed: 5 of 12 images kept back, none of them in the training set, the same 5 every time")


def check_generated_labels() -> None:
    """labelgen.py: a seed always gives the same tile; training and held-out seeds never meet."""
    import labelgen
    from data import GeneratedLabelDataset, data_source, label_tile, make_dataset

    for seed in (0, 7, 1_234_567):
        first, second = labelgen.render_tile(seed), labelgen.render_tile(seed)
        assert first.shape == (TILE_PX, TILE_PX, 3) and first.dtype == np.uint8, f"seed {seed}: shape {first.shape}"
        assert np.array_equal(first, second), f"seed {seed} gave two different tiles"
    assert not np.array_equal(labelgen.render_tile(0), labelgen.render_tile(1)), "seeds 0 and 1 gave the same tile"

    train, held = labelgen.TRAIN_SEEDS, labelgen.HELDOUT_SEEDS
    assert train.step == 1 and held.step == 1, "seed ranges must be contiguous"
    assert max(train.start, held.start) >= min(train.stop, held.stop), f"training seeds {train} overlap held-out seeds {held}"
    for name, seeds in (("acceptance", labelgen.ACCEPTANCE_SEEDS), ("spread check", labelgen.SPREAD_SEEDS)):
        assert held.start <= seeds.start and seeds.stop <= held.stop and len(seeds) == 256, f"{name} seeds {seeds} are not 256 held-out seeds"
    assert not set(labelgen.ACCEPTANCE_SEEDS) & set(labelgen.SPREAD_SEEDS), "acceptance and spread-check seeds overlap"

    dataset = make_dataset({"data": {"source": "generated_labels"}})
    assert isinstance(dataset, GeneratedLabelDataset) and dataset.seeds == train, "the generated source does not use the training seeds"
    tile = dataset[0]
    assert tile.shape == (3, TILE_PX, TILE_PX) and 0.0 <= float(tile.min()) and float(tile.max()) <= 1.0, "a generated tile is not (3, 384, 384) in [0,1]"
    assert torch.equal(tile, label_tile(train.start)), "dataset item 0 is not the tile of the first training seed"
    try:
        GeneratedLabelDataset(range(held.stop - 10, held.stop + 10))
    except ValueError:
        pass
    else:
        raise AssertionError("a training set reaching into the held-out seeds was accepted")
    assert data_source({"data": {}}) == "folder", "a config without data.source no longer reads the folder"
    try:
        data_source({"data": {"source": "photos"}})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown data.source was accepted")
    print(
        f"generated-label checks passed: the same seed gives the same tile; training seeds {train.start:,} to "
        f"{train.stop - 1:,} and held-out seeds {held.start:,} to {held.stop - 1:,} do not overlap"
    )


def check_rotation_tag() -> None:
    """Photos load upright: the rotation tag a phone writes (EXIF orientation) is applied."""
    stored = np.zeros((20, 40, 3), dtype=np.uint8)  # 40 wide, 20 high as stored
    stored[:10, :10] = (255, 0, 0)  # red at the stored top left
    with tempfile.TemporaryDirectory(prefix="tilemark_rotation_") as tmp:
        tag = Image.Exif()
        tag[0x0112] = 6  # "turn a quarter clockwise to show": the stored top left belongs at the top right
        Image.fromarray(stored).save(Path(tmp) / "turned.png", exif=tag)
        Image.fromarray(stored).save(Path(tmp) / "plain.png")
        turned = load_rgb(Path(tmp) / "turned.png")
        plain = load_rgb(Path(tmp) / "plain.png")
        dataset = ImageFolderDataset(tmp, image_size=16)
        sizes = {path.name: dataset.load(index).size for index, path in enumerate(dataset.files)}
    assert turned.size == (20, 40), f"a photo tagged as turned loaded as {turned.size}, wanted (20, 40)"
    assert turned.getpixel((15, 5)) == (255, 0, 0) and turned.getpixel((5, 5)) == (0, 0, 0), "turned the wrong way"
    assert plain.size == (40, 20) and plain.getpixel((5, 5)) == (255, 0, 0), "a photo without the tag was changed"
    assert sizes == {"plain.png": (40, 20), "turned.png": (20, 40)}, f"the dataset does not load upright: {sizes}"
    print("rotation checks passed: a photo tagged as turned loads upright; one without the tag is untouched")


def check_multi_shot() -> None:
    """acceptance.vote: K shots are read by averaging the reader's logits over the first K, then taking signs."""
    from acceptance import vote

    bits = torch.stack([torch.ones(MESSAGE_BITS), (torch.arange(MESSAGE_BITS) % 2).float()])
    truth = bits * 2.0 - 1.0  # +1 for a 1, -1 for a 0
    logits = truth.expand(9, -1, -1).clone()
    logits[0, 0, :10] = -0.5  # shot 1 gets 10 bits of tile 1 wrong, weakly
    logits[8, 1] = -10.0 * truth[1]  # shot 9 gets every bit of tile 2 wrong, strongly
    rows = {row["shots"]: row for row in vote(logits, bits, (1, 3, 5, 9))}
    assert rows[1]["decoded"] == 0.5 and abs(rows[1]["bit_accuracy"] - (1 - 10 / 128)) < 1e-6, f"one shot: {rows[1]}"
    assert rows[3]["decoded"] == 1.0 and rows[3]["all_64_right"] == 1.0, f"three shots did not outvote the weak errors: {rows[3]}"
    assert rows[5]["decoded"] == 1.0, f"five shots reached the ninth: {rows[5]}"
    assert rows[9]["decoded"] == 0.5 and rows[9]["bit_accuracy"] == 0.5, f"nine shots did not include the ninth: {rows[9]}"
    try:
        vote(logits[:3], bits, (1, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("asking for 5 shots of 3 was accepted")

    # The visibility pass line: share of tiles with p95 at most the target, flat tiles apart; none without a target.
    from acceptance import MIN_UNDER_TARGET, visibility_summary

    p95 = torch.tensor([0.5] * 19 + [3.0])
    flat = torch.tensor([False] * 16 + [True] * 4)
    seen = visibility_summary(p95, flat, 2.0)
    assert seen["under_target"] == 0.95 and seen["visibility_pass"] is (0.95 >= MIN_UNDER_TARGET), seen
    assert seen["under_target_flat"] == 0.75 and seen["flat_tiles"] == 4, seen
    assert not visibility_summary(p95, flat, 0.4)["visibility_pass"], "no tile under 0.4, yet a pass"
    seen = visibility_summary(p95, flat, None)
    assert seen["visibility_pass"] is None and seen["under_target"] is None, f"a pass line without a target: {seen}"
    assert seen["visibility_p95_median"] == 0.5 and seen["visibility_flat_p95_median"] == 0.5, seen
    print(
        "multi-shot checks passed: averaged logits outvote a weak wrong shot; K uses exactly the first K shots; "
        "visibility pass line counts tiles under the target, flat tiles apart, none without a target"
    )


def check_scan(trainer: Trainer) -> None:
    """simulate_scan: the frames of one scan share pose, blur, colour and camera resolution; noise and JPEG are per frame.

    On the CPU with every effect on, at full strength. Also: one frame of a
    scan is simulate itself (same draws in the same order), and over 256
    samples it draws blur, colour and resolution from the same distributions.
    """
    import simulate as sim_module
    from simulate import simulate_scan

    cfg = copy.deepcopy(trainer.cfg)
    cfg["simulation"]["enabled"] = True
    cfg["simulation"]["effects"] = {name: True for name in cfg["simulation"]["effects"]}
    assert cfg.get("camera_px_per_tile") is not None, "check_scan needs camera_px_per_tile for the resolution step"
    generator = torch.Generator().manual_seed(0)
    coarse = torch.rand(2, 3, 12, 12, generator=generator)
    surface = F.interpolate(coarse, size=(3 * TILE_PX, 3 * TILE_PX), mode="bicubic", align_corners=False).clamp(0.0, 1.0)

    torch.manual_seed(0)
    scan = simulate_scan(surface, cfg, 1.0, frames=3)
    assert len(scan) == 3, f"3 frames asked for, {len(scan)} given"
    for i, shot in enumerate(scan[1:], start=2):
        assert torch.equal(shot.homography, scan[0].homography), f"frame {i} has its own pose"
        gap = (shot.frame - scan[0].frame).abs().amax(dim=(1, 2, 3)).min().item()
        assert gap > 1e-3, f"frame {i} is frame 1 again (largest difference {gap:.2e}) with noise and JPEG on"

    quiet = copy.deepcopy(cfg)
    quiet["simulation"]["effects"]["noise"] = quiet["simulation"]["effects"]["jpeg"] = False
    torch.manual_seed(0)
    scan = simulate_scan(surface, quiet, 1.0, frames=3)
    for i, shot in enumerate(scan[1:], start=2):
        assert torch.equal(shot.frame, scan[0].frame), "with noise and JPEG off, the frames of a scan differ"

    torch.manual_seed(1)
    single = simulate(surface, cfg, 1.0)
    torch.manual_seed(1)
    (first,) = simulate_scan(surface, cfg, 1.0, frames=1)
    assert torch.equal(first.homography, single.homography), "a one-frame scan does not draw the pose simulate draws"
    assert torch.equal(first.frame, single.frame), "a one-frame scan is not simulate under the same seed"

    # Distributions over 256 samples, with different seeds: perspective off and
    # a small surface keep it cheap. Each draw is recorded as it is made.
    draws = copy.deepcopy(cfg)
    draws["simulation"]["effects"]["perspective"] = False
    small = torch.rand(256, 3, 32, 32, generator=generator)
    originals = {name: getattr(sim_module, name) for name in ("blur_kernels", "colour_draws", "_camera_ratios")}

    def recorded(run) -> dict[str, torch.Tensor]:
        seen: dict[str, list[torch.Tensor]] = {"defocus": [], "motion": [], "colour": [], "ratio": []}

        def blur_kernels(defocus, motion, angle, size):
            seen["defocus"].append(defocus.detach().double().cpu())
            seen["motion"].append(motion.detach().double().cpu())
            return originals["blur_kernels"](defocus, motion, angle, size)

        def colour_draws(*args, **kwargs):
            out = originals["colour_draws"](*args, **kwargs)
            seen["colour"].append(torch.cat([part.detach().double().cpu().flatten(1) for part in out], dim=1))
            return out

        def camera_ratios(*args, **kwargs):
            out = originals["_camera_ratios"](*args, **kwargs)
            seen["ratio"].append(torch.tensor(out, dtype=torch.float64))
            return out

        sim_module.blur_kernels, sim_module.colour_draws, sim_module._camera_ratios = blur_kernels, colour_draws, camera_ratios
        try:
            run()
        finally:
            for name, function in originals.items():
                setattr(sim_module, name, function)
        return {name: torch.cat(values) for name, values in seen.items()}

    torch.manual_seed(2)
    a = recorded(lambda: simulate(small, draws, 1.0))
    torch.manual_seed(3)
    b = recorded(lambda: simulate_scan(small, draws, 1.0, frames=1))
    for name in a:
        assert a[name].shape == b[name].shape == (256, *a[name].shape[1:]), f"{name}: {a[name].shape} and {b[name].shape} draws"
        error = (a[name].var(dim=0) / 256 + b[name].var(dim=0) / 256).sqrt()
        off = ((a[name].mean(dim=0) - b[name].mean(dim=0)).abs() / error.clamp_min(1e-12)).max().item()
        assert off < 4.5, f"{name}: a one-frame scan draws {off:.1f} standard errors away from simulate"
    print(
        "scan checks passed: frames share the pose exactly, differ only by noise and JPEG, one frame is simulate, "
        "blur, colour and resolution drawn alike"
    )


def check_locator_view() -> None:
    """Stage 2: in-plane rotation, a print that ends inside the frame, the target corners and the turn-free loss.

    Geometry only (every effect but perspective off), float64 on the CPU.
    """
    from locator_train import corner_loss, turned
    from simulate import sample_homography, simulate_view, target_corners

    cfg = load_config(HERE / "configs" / "locator.yaml")
    cfg["simulation"]["effects"] = {name: name == "perspective" for name in cfg["simulation"]["effects"]}
    torch.manual_seed(0)

    # 1. Rotation: off gives the surface's x axis along the frame's x axis; 360 spreads it over every angle.
    def angles(deg: float) -> torch.Tensor:
        geo = copy.deepcopy(cfg)
        geo["simulation"]["perspective"]["rotation_deg"] = deg
        geo["simulation"]["perspective"]["max_tilt_deg"] = 0.0
        h, _ = sample_homography(512, TILE_PX, geo, 1.0, tile_px=(100.0, 100.0), frame_wh=(384, 384))
        return torch.atan2(h[:, 1, 0], h[:, 0, 0])

    still, turning = angles(0.0), angles(360.0)
    assert still.abs().max() < 1e-9, f"rotation_deg 0 still rotates, by up to {still.abs().max():.2e} rad"
    quarters = torch.histc(torch.remainder(turning, 2 * np.pi), bins=4, min=0.0, max=2 * np.pi)
    assert quarters.min() > 80, f"rotation_deg 360 does not cover every quarter turn: {quarters.tolist()}"

    # 2. Target corners: flattening the frame with them gives the tile-sized square of the surface
    #    centred on the point at the middle of the frame, at whatever rotation and tilt.
    generator = torch.Generator().manual_seed(1)
    coarse = torch.rand(4, 3, 6, 6, generator=generator, dtype=torch.float64)
    smooth = F.interpolate(coarse.repeat(1, 1, 3, 3), scale_factor=TILE_PX / 6, mode="bicubic", align_corners=False)
    tile = smooth[..., TILE_PX : 2 * TILE_PX, TILE_PX : 2 * TILE_PX].clamp(0.0, 1.0).contiguous()
    geo = copy.deepcopy(cfg)
    geo["simulation"]["surface_extent_tiles"] = None
    view = simulate_view(tile, geo, 1.0, (768, 768), (330.0, 380.0))
    assert torch.allclose(view.corners, target_corners(view.homography, 768, 768)), "returned corners are not the target"
    patch = flatten_from_corners(view.frame, view.corners, TILE_PX)
    on_surface = apply_homography(torch.linalg.inv(view.homography), view.corners)
    straight = warp_periodic(tile, _square_to(on_surface), (TILE_PX, TILE_PX))
    gap = (patch - straight)[..., 8:-8, 8:-8].abs().max().item()
    assert gap < 0.02, f"the target square flattens to the wrong part of the surface (off by {gap:.3f})"
    at_middle = apply_homography(torch.linalg.inv(view.homography), torch.full((4, 1, 2), 384.0, dtype=torch.float64))
    off_centre = (on_surface.mean(dim=1, keepdim=True) - at_middle).norm(dim=-1).max().item()
    assert off_centre < 1e-6, f"the target square is not centred on the point at the middle of the frame ({off_centre:.3f} px off)"
    side = (on_surface[:, 1] - on_surface[:, 0]).norm(dim=-1)
    assert torch.allclose(side, torch.full_like(side, float(TILE_PX)), atol=1e-6), f"target square sides {side.tolist()}"

    # 3. The print ends: a flat tile on a background of another colour. The target square is all print,
    #    and with the edges 0.6 to 1.0 tiles away every frame shows background too.
    geo = copy.deepcopy(cfg)
    geo["simulation"]["surface_extent_tiles"] = [0.6, 1.0]
    flat = torch.full((8, 3, TILE_PX, TILE_PX), 0.25, dtype=torch.float64)
    background = torch.full((8, 3, TILE_PX, TILE_PX), 0.9, dtype=torch.float64)
    view = simulate_view(flat, geo, 1.0, (384, 384), (55.0, 121.0), background)
    print_px = ((view.frame - 0.25).abs().amax(dim=1) < 1e-6).float()
    square = flatten_from_corners(print_px.unsqueeze(1), view.corners, 64)[..., 4:-4, 4:-4]
    assert square.min() > 0.99, "background inside the target square"
    shown = print_px.mean(dim=(1, 2))
    assert shown.max() < 0.9, f"the print fills the frame although it ends within a tile: {shown.tolist()}"
    endless = simulate_view(flat, cfg | {"simulation": {**cfg["simulation"], "surface_extent_tiles": None}}, 1.0, (384, 384), (55.0, 121.0))
    assert ((endless.frame - 0.25).abs() < 1e-6).all(), "surface_extent_tiles null still shows background"

    # 4. The loss takes any of the four turns of the target, and nothing else.
    target = view.corners
    size = view.tile_px
    for r in range(4):
        assert corner_loss(turned(target, r), target, size, 3.0).item() < 1e-9, f"turn {r} is not free"
    mirrored = target[:, [1, 0, 3, 2]]
    assert corner_loss(mirrored, target, size, 3.0).item() > 10.0, "a mirrored square costs nothing"
    print(
        "locator view checks passed: rotation off and over every quarter turn, target square is the tile-sized "
        "square at the middle, the print ends with background beyond it, all four turns free in the loss"
    )


def _square_to(corners: torch.Tensor) -> torch.Tensor:
    """Homography from the TILE_PX output square to ``corners`` (B,4,2)."""
    from geometry import homography_from_points

    return homography_from_points(square_corners(float(TILE_PX), corners.shape[0], dtype=corners.dtype), corners)


def check_reading_check(device_name: str) -> None:
    """reading_check: a real short train.py run stops with exit code 7 and a checkpoint, and will not resume."""
    import json
    import subprocess
    import sys

    with tempfile.TemporaryDirectory(prefix="tilemark_reading_") as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        data_dir.mkdir()
        make_tiny_dataset(data_dir)
        command = [
            sys.executable, str(HERE / "train.py"), "--config", str(HERE / "configs" / "smoke.yaml"),
            f"data.train_dir={data_dir.as_posix()}", f"checkpoint.dir={(root / 'ckpt').as_posix()}",
            f"log.dir={(root / 'logs').as_posix()}", f"train.device={device_name}",
            "train.total_steps=20", "log.report_every=10", "log.spread_images=0",
            "reading_check.enabled=true", "reading_check.at_step=10", "reading_check.min_bit_acc=0.99",
        ]
        first = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
        assert first.returncode == 7, f"reading check: exit code {first.returncode}, expected 7\n{first.stdout[-1500:]}\n{first.stderr[-1500:]}"
        assert "reading too weak" in first.stdout, first.stdout[-1500:]
        assert list((root / "ckpt").glob("*")), "reading check: no checkpoint was saved"
        state = json.loads((root / "logs" / "run_state.json").read_text(encoding="utf-8"))
        assert state["reading_check"]["result"] == "failed", state["reading_check"]
        second = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
        assert second.returncode == 7 and "will not resume" in second.stdout, second.stdout[-1500:]
    print("reading check passed: exit code 7, checkpoint saved, run_state.json says failed, no resume")


def describe_simulation(cfg: dict) -> str:
    sim = cfg["simulation"]
    if not sim["enabled"]:
        return "simulation off"
    effects = [name for name, on in sim["effects"].items() if on]
    return f"simulation on ({', '.join(effects)}) at strength {cfg['schedule']['sim_strength']['max_strength']}"


def run_smoke(
    device_name: str | None = None,
    overrides: list[str] | None = None,
    config: str | Path | None = None,
    sheet_path: str | Path | None = None,
) -> dict[str, float]:
    device = torch.device(device_name or default_device())
    config = Path(config) if config else HERE / "configs" / "smoke.yaml"
    with tempfile.TemporaryDirectory(prefix="tilemark_smoke_") as tmp:
        make_tiny_dataset(Path(tmp))
        cfg = load_config(config, [f"data.train_dir={Path(tmp).as_posix()}", *(overrides or [])])
        torch.manual_seed(cfg["train"]["seed"])
        trainer = Trainer(cfg, device)
        steps = cfg["train"]["total_steps"]
        every = cfg["log"]["print_every"]
        loader = make_loader(cfg, device)
        print(
            f"smoke test: {steps} steps, batch {cfg['train']['batch_size']}, {len(loader.dataset)} images, "
            f"device={device}, {describe_simulation(cfg)}, reader_input={cfg['train']['reader_input']}"
        )
        print(trainer.models.band.describe())
        print(
            f"MESSAGE_BITS={MESSAGE_BITS}  TILE_PX={TILE_PX}  "
            f"embedder {count_parameters(trainer.embedder) / 1e6:.2f}M params, "
            f"reader {count_parameters(trainer.reader) / 1e6:.2f}M params"
        )
        check_structure(trainer)
        print(
            "structure checks passed: circular convs, seamless residual, reader under 3M, exact flatten, "
            "portrait, landscape and square frames, periodic sampling"
        )
        check_spread(trainer)
        check_texture_mask(trainer)
        check_visibility()
        check_follow_accuracy(trainer)
        check_follow_visibility(trainer)
        check_visibility_step(trainer)
        check_calibration_levels()
        check_center_fraction()
        check_rotation_tag()
        check_holdout()
        check_generated_labels()
        check_multi_shot()
        check_scan(trainer)
        check_locator_view()
        check_reading_check(str(device))

        batches = infinite(loader)
        history: list[dict[str, float]] = []
        t0 = time.time()
        while trainer.step < steps:
            history.append(trainer.train_step(next(batches)))
            if trainer.step % every == 0:
                window = history[-every:]
                acc = sum(m["bit_acc"] for m in window) / len(window)
                last = history[-1]
                print(
                    f"step {trainer.step:4d}  bit_acc {acc:.3f}  bit_loss {last['bit_loss']:.4f}  "
                    f"psnr {last['psnr']:5.1f} dB  image_scale {last['image_scale']:.3f}  "
                    f"{time.time() - t0:5.0f}s",
                    flush=True,
                )

        first = sum(m["bit_acc"] for m in history[:every]) / every
        final = sum(m["bit_acc"] for m in history[-every:]) / every
        evals = [trainer.evaluate(next(batches), keep_outputs=True) for _ in range(8)]
        eval_acc = sum(m["bit_acc"] for m in evals) / len(evals)
        eval_exact = sum(m["message_exact"] for m in evals) / len(evals)
        eval_correctable = sum(m["message_correctable"] for m in evals) / len(evals)
        eval_psnr = sum(m["psnr"] for m in evals) / len(evals)
        sheet = sample_sheet(evals[0]["outputs"]["original"], evals[0]["outputs"]["marked"])
        sheet_path = save_image(sheet, sheet_path or HERE / "outputs" / "smoke_samples.png", footer=cfg["data"].get("credit"))

    print(f"\nfirst {every} steps:  bit_acc {first:.3f}")
    print(f"last {every} steps:   bit_acc {final:.3f}")
    print(
        f"eval mode, fresh bits and offsets: bit_acc {eval_acc:.3f}, "
        f"all 64 bits right {eval_exact:.3f}, within BCH reach {eval_correctable:.3f}"
    )
    print(f"PSNR marked vs original: {eval_psnr:.1f} dB (image-loss weight at the end: {history[-1]['image_scale']:.3f})")
    print(f"sample sheet (original | marked | residual x10): {sheet_path}")
    print(f"total time: {time.time() - t0:.0f}s")
    return {"first": first, "final": final, "eval": eval_acc, "psnr": eval_psnr}


def check_result(result: dict[str, float]) -> None:
    assert result["final"] >= REQUIRED_ACCURACY, (
        f"FAIL: training bit accuracy {result['final']:.3f} < {REQUIRED_ACCURACY}"
    )
    assert result["eval"] >= REQUIRED_ACCURACY, f"FAIL: eval bit accuracy {result['eval']:.3f} < {REQUIRED_ACCURACY}"


def test_smoke() -> None:
    check_result(run_smoke())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first of those available)")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides, for experiments")
    parser.add_argument("--config", default=None, help="config file (default: configs/smoke.yaml)")
    parser.add_argument("--sheet", default=None, help="where to save the sample sheet (default: outputs/smoke_samples.png)")
    args = parser.parse_args()
    result = run_smoke(args.device, args.overrides, args.config, args.sheet)
    check_result(result)
    print(f"PASS: bit accuracy {result['final']:.3f} (train) / {result['eval']:.3f} (eval) >= {REQUIRED_ACCURACY}")
