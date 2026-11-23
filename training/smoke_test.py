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
from geometry import apply_homography, flatten_from_corners, flatten_random_patch, square_corners, warp_periodic
from models import count_parameters
from simulate import simulate
from spread import concentration, hide_mask, hide_regions, hide_settings, local_energy, peak_penalty, peak_settings
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
    print(
        f"spread checks passed: bit-loss gradient norm {to_embedder:.3g} at the embedder and {to_reader:.3g} at the "
        f"reader with {out['hidden_fraction'] * 100:.0f} percent of the patch blanked; "
        f"peak-penalty gradient norm {norm(from_peak):.3g} at the embedder"
    )


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
    print("multi-shot checks passed: averaged logits outvote a weak wrong shot; K uses exactly the first K shots")


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
        check_follow_accuracy(trainer)
        check_center_fraction()
        check_rotation_tag()
        check_holdout()
        check_generated_labels()
        check_multi_shot()
        check_locator_view()

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
