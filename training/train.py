"""Stage 1 training: embedder and reader.

    python train.py --config configs/default.yaml data.train_dir=/path/to/images

Resumes automatically from the newest checkpoint in ``checkpoint.dir``.
Pass --fresh to start over; existing checkpoints are moved aside, not deleted.

Overrides may come before or after the options.

Every ``log.report_every`` steps one line goes to the console and to
``<log.dir>/progress.csv``: step, bit accuracy, PSNR, the image-loss weight
and, with ``--price-per-hour``, the cost so far.

    python train.py --config configs/runpod_direct.yaml --price-per-hour 0.69

Ctrl-C or SIGTERM saves a checkpoint and exits. Exit codes: 0 finished or
stopped by ``--max-minutes``, 3 stopped by the learning check
(``early_stop``), 4 stopped because the mark piled up at a fixed place in the
tile (``spread.stop_above``),
5 stopped because the image loss never switched on (``image_loss_check``),
6 stopped at its limit on hours or cost (``budget``), 7 stopped because
reading was too weak at the reading check (``reading_check``), 130 interrupted.

With every sample sheet the run measures where the mark sits on 256 fixed
dataset images (the spread check) and appends a row to ``<log.dir>/spread.csv``.

When the training loop ends, for any reason other than a crash,
``<log.dir>/throughput.json`` records this process's steps, seconds and how
long the loop waited for the data loader (bench_report.py reads it).

On a CUDA device the run first prints what the GPU is and how much memory is
free. ``train.expect_gpu`` names the GPU the run must be on; anything else,
including no CUDA at all, stops the run. It never falls back to the CPU then.

    python train.py --config configs/sanity.yaml --max-minutes 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from camera import camera_range, check_demo
from checkpoint import load_latest_checkpoint, save_checkpoint, set_aside_checkpoints
from config import load_config
from constants import MESSAGE_BITS, TILE_PX
from data import infinite, make_loader
from geometry import center_crop, flatten_random_patch
from losses import texture_settings, texture_weights
from models import build_models, count_parameters
from runlog import ProgressLog, RunClock, SpreadLog
from simulate import simulate, wrap_surface
from visibility import batch_statistic, tile_visibility, visibility_settings
from spread import (
    busy_windows,
    concentration,
    fixed_batch_for,
    hide_regions,
    hide_settings,
    peak_penalty,
    peak_settings,
    spread_report,
)
from viz import add_footer, residual_image, sample_sheet, save_image

READER_INPUTS = ("flattened", "direct")
# BCH(63,39) in /docs/message-format.md corrects up to 4 errors in bits 0-62.
BCH_CODEWORD_BITS = 63
BCH_CORRECTABLE = 4


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def bit_metrics(logits: torch.Tensor, bits: torch.Tensor) -> dict[str, float]:
    """Bit accuracy and the two full-message rates for a batch."""
    wrong = (logits > 0).float() != bits
    codeword_errors = wrong[:, :BCH_CODEWORD_BITS].sum(dim=1)
    return {
        "bit_acc": 1.0 - wrong.float().mean().item(),
        # full-message decode rate without error correction: all 64 bits right
        "message_exact": (wrong.sum(dim=1) == 0).float().mean().item(),
        # full-message decode rate with it: at most 4 errors in the 63-bit codeword
        "message_correctable": (codeword_errors <= BCH_CORRECTABLE).float().mean().item(),
    }


class Trainer:
    """Owns the models, the optimizer and the two schedules.

    Schedules (config section ``schedule``):

    * ``image_loss``: the image-loss weight is 0 until the smoothed bit
      accuracy first passes ``acc_threshold``, then ramps linearly to 1 over
      ``ramp_steps`` steps. With ``follow_accuracy: {floor, band}`` set, the
      weight instead moves by 1/``ramp_steps`` per step after that trigger:
      up while the smoothed bit accuracy is at or above ``floor``, down while
      it is below ``floor - band``, unchanged in between, never outside 0 to 1.
    * ``sim_strength``: the simulation strength is 0 until ``start_step``,
      then ramps linearly to ``max_strength`` over ``ramp_steps`` steps
      (0 steps: full strength from ``start_step``).

    Config section ``spread`` (see spread.py), both off by default: the
    ``peak_penalty`` loss follows the image-loss schedule, and ``hide``
    blanks regions of the reader's patch in training steps only.
    """

    def __init__(self, cfg: dict, device: torch.device):
        if cfg["train"]["stage"] != 1:
            raise NotImplementedError("only stage 1 (embedder + reader) is written")
        if cfg["train"]["reader_input"] not in READER_INPUTS:
            raise ValueError(f"train.reader_input must be one of {READER_INPUTS}")
        self.cfg = cfg
        self.device = device
        self.models = build_models(cfg)
        for module in self.models.named().values():
            module.to(device)
        self.embedder, self.reader = self.models.embedder, self.models.reader
        self.params = list(self.embedder.parameters()) + list(self.reader.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg["train"]["lr"])
        # Built up front so a missing package or weight file fails at start, not mid-run.
        self.perceptual = None
        if cfg["loss"]["lpips_weight"] > 0:
            from losses import PerceptualLoss

            self.perceptual = PerceptualLoss(cfg["loss"]["lpips_net"]).to(device)
        self.step = 0
        self.acc_ema = 0.5
        self.image_ramp_start: int | None = None
        # The image-loss weight while schedule.image_loss.follow_accuracy is set; unused otherwise.
        self.follow_scale = 0.0
        # The visibility-loss weight and the smoothed visibility while schedule.image_loss.follow_visibility is set.
        self.vis_weight = 0.01
        self.vis_ema: float | None = None
        # The controller's window: the last `window` per-tile p95 values, across steps.
        self.vis_window: deque[float] = deque(maxlen=256)
        self.follow_settings()  # a bad setting fails at start, not at the trigger
        follow_visibility = self.visibility_follow_settings()
        if follow_visibility is not None:
            if self.follow_settings() is not None:
                raise ValueError("schedule.image_loss.follow_accuracy and follow_visibility cannot both be set")
            if not visibility_settings(cfg)["enabled"]:
                raise ValueError(
                    "schedule.image_loss.follow_visibility is set but loss.visibility.enabled is false: the controller "
                    "would have nothing to follow. Set loss.visibility.enabled: true, or follow_visibility: null."
                )
            self.vis_weight = follow_visibility["init_weight"]
            self.vis_window = deque(maxlen=follow_visibility["window"])
        texture_settings(cfg)  # likewise for loss.texture_mask

    # -- schedules ---------------------------------------------------------

    def follow_settings(self) -> dict | None:
        """``schedule.image_loss.follow_accuracy`` as {floor, band}, or None when it is not set."""
        follow = self.cfg["schedule"]["image_loss"].get("follow_accuracy")
        if not follow:
            return None
        floor, band = float(follow["floor"]), float(follow.get("band", 0.0))
        if not 0.0 <= floor <= 1.0 or band < 0.0:
            raise ValueError(f"schedule.image_loss.follow_accuracy needs floor in 0 to 1 and band at least 0, got {follow}")
        return {"floor": floor, "band": band}

    def visibility_follow_settings(self) -> dict | None:
        """``schedule.image_loss.follow_visibility`` with its defaults filled in, or None when it is not set."""
        follow = self.cfg["schedule"]["image_loss"].get("follow_visibility")
        if not follow:
            return None
        if follow.get("target") is None:
            raise ValueError(f"schedule.image_loss.follow_visibility needs a target, got {follow}")
        settings = {
            "target": float(follow["target"]),
            "quantile": float(follow.get("quantile", 0.96)),
            "window": int(follow.get("window", 256)),
            "ema": float(follow.get("ema", 0.98)),
            "low": float(follow.get("low", 0.9)),
            "rate": float(follow.get("rate", 4.6e-4)),
            "init_weight": float(follow.get("init_weight", 0.01)),
            "min_weight": float(follow.get("min_weight", 0.01)),
            "max_weight": float(follow.get("max_weight", 1000.0)),
        }
        ok = (
            settings["target"] > 0.0
            and 0.0 < settings["quantile"] <= 1.0
            and settings["window"] >= 1
            and 0.0 <= settings["ema"] < 1.0
            and 0.0 < settings["low"] <= 1.0
            and settings["rate"] > 0.0
            and 0.0 < settings["min_weight"] <= settings["init_weight"] <= settings["max_weight"]
        )
        if not ok:
            raise ValueError(
                "schedule.image_loss.follow_visibility needs target > 0, quantile and low in (0, 1], window >= 1, ema in [0, 1), "
                f"rate > 0 and 0 < min_weight <= init_weight <= max_weight, got {follow}"
            )
        return settings

    def observe_visibility(self, tile_values) -> float:
        """Add a step's per-tile p95 masked delta E values to the controller's window; return the window's quantile.

        The window holds the last ``window`` per-tile values across steps (fewer until it has filled), so its
        ``quantile`` (0.96) is a population percentile of the tiles the model sees: the share of tiles at or under
        the target that acceptance.py checks. A batch quantile over 8 tiles would sit near the 81st-86th percentile.
        Non-finite values are left out; NaN while the window is empty. Called before and after the trigger.
        """
        follow = self.visibility_follow_settings()
        quantile = 0.96 if follow is None else follow["quantile"]
        for value in tile_values:
            value = float(value)
            if math.isfinite(value):
                self.vis_window.append(value)
        if not self.vis_window:
            return math.nan
        return torch.quantile(torch.tensor(list(self.vis_window), dtype=torch.float64), quantile).item()

    def update_visibility(self, value: float) -> None:
        """After a training step: smooth the window's visibility (observe_visibility) and move the visibility-loss weight.

        Does nothing before the image-loss trigger. The weight is multiplied by 1 + rate while the smoothed value is
        above the target and divided by it while below ``low`` times the target, within min_weight to max_weight.
        """
        follow = self.visibility_follow_settings()
        if follow is None or self.image_ramp_start is None:
            return
        if self.vis_ema is None:
            self.vis_ema = float(value)
        else:
            self.vis_ema = follow["ema"] * self.vis_ema + (1.0 - follow["ema"]) * float(value)
        if self.vis_ema > follow["target"]:
            self.vis_weight = min(follow["max_weight"], self.vis_weight * (1.0 + follow["rate"]))
        elif self.vis_ema < follow["low"] * follow["target"]:
            self.vis_weight = max(follow["min_weight"], self.vis_weight / (1.0 + follow["rate"]))

    def visibility_quantile(self) -> float:
        """The batch quantile of the tiles' visibility in evaluate, and in the logs when there is no controller:
        the controller's quantile, 0.9 without one. With the controller the logs show observe_visibility's reading."""
        follow = self.visibility_follow_settings()
        return 0.9 if follow is None else float(follow["quantile"])

    def ramp_scale(self) -> float:
        """The plain schedule: a linear ramp from the trigger step."""
        if self.image_ramp_start is None:
            return 0.0
        steps = int(self.cfg["schedule"]["image_loss"]["ramp_steps"])
        if steps <= 0:
            return 1.0
        return min(1.0, (self.step - self.image_ramp_start) / steps)

    def image_loss_scale(self) -> float:
        """Weight of the image loss, and of the peak penalty, in the next training step: 0 to 1."""
        if self.image_ramp_start is None:
            return 0.0
        if self.follow_settings() is not None:
            return self.follow_scale
        return self.ramp_scale()

    def update_schedule(self, bit_acc: float) -> None:
        """After a training step: smooth the bit accuracy, trigger the image loss, move its weight."""
        sched = self.cfg["schedule"]["image_loss"]
        self.acc_ema = sched["acc_ema"] * self.acc_ema + (1.0 - sched["acc_ema"]) * bit_acc
        follow = self.follow_settings()
        steps = int(sched["ramp_steps"])
        if self.image_ramp_start is None:
            if self.acc_ema > sched["acc_threshold"]:
                self.image_ramp_start = self.step
                # Like the plain ramp: nothing on the trigger step unless there is no ramp at all.
                self.follow_scale = 1.0 if steps <= 0 else 0.0
            return
        if follow is None:
            return
        move = 1.0 if steps <= 0 else 1.0 / steps
        if self.acc_ema >= follow["floor"]:
            self.follow_scale = min(1.0, self.follow_scale + move)
        elif self.acc_ema < follow["floor"] - follow["band"]:
            self.follow_scale = max(0.0, self.follow_scale - move)

    def sim_strength(self) -> float:
        if not self.cfg["simulation"]["enabled"]:
            return 0.0
        sched = self.cfg["schedule"]["sim_strength"]
        since_start = self.step - int(sched["start_step"])
        if since_start < 0:
            return 0.0
        steps = int(sched["ramp_steps"])
        progress = 1.0 if steps <= 0 else min(1.0, since_start / steps)
        return float(sched["max_strength"]) * progress

    # -- forward pass shared by training and evaluation --------------------

    def reader_view(self, frame: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        """What the reader is given. The only place the two reader_input modes differ."""
        if self.cfg["train"]["reader_input"] == "flattened":
            patch, _ = flatten_random_patch(
                frame,
                homography,
                tile_px=TILE_PX,
                corner_jitter_px=float(self.cfg["flatten"]["corner_jitter_px"]),
            )
            return patch
        return center_crop(frame, TILE_PX)

    def forward(self, tiles: torch.Tensor, bits: torch.Tensor, hide: bool = False) -> dict:
        """``hide``: blank random regions of the reader's patch, if spread.hide is on."""
        residual = self.embedder(tiles, bits)
        marked = (tiles + residual).clamp(0.0, 1.0)
        direct = self.cfg["train"]["reader_input"] == "direct"
        # Direct mode samples the tile as a repeating pattern, so it needs no repeats.
        surface = marked if direct else wrap_surface(marked, int(self.cfg["surface"]["repeat"]))
        sim = simulate(surface, self.cfg, self.sim_strength(), direct=direct)
        view = self.reader_view(sim.frame, sim.homography)
        hidden = 0.0
        if hide and hide_settings(self.cfg)["enabled"]:
            view, hidden = hide_regions(view, self.cfg)
        logits = self.reader(view)
        return {
            "residual": residual,
            "marked": marked,
            "frame": sim.frame,
            "view": view,
            "logits": logits,
            "hidden_fraction": hidden,
        }

    def random_bits(self, batch: int) -> torch.Tensor:
        return torch.randint(0, 2, (batch, MESSAGE_BITS), device=self.device).float()

    # -- one optimisation step ---------------------------------------------

    def train_step(self, tiles: torch.Tensor, keep_outputs: bool = False) -> dict:
        weights = self.cfg["loss"]
        self.embedder.train()
        self.reader.train()
        tiles = tiles.to(self.device, non_blocking=True)
        bits = self.random_bits(tiles.shape[0])

        out = self.forward(tiles, bits, hide=True)
        bit_loss = F.binary_cross_entropy_with_logits(out["logits"], bits)
        # L2 on the residual before the [0,1] clamp, so pixels pushed past
        # the clamp are still pulled back.
        l2_loss = out["residual"].pow(2).mean()
        image_scale = self.image_loss_scale()
        loss = weights["bit_weight"] * bit_loss
        lpips_loss = None
        peak_loss = None
        l2_texture_loss = None
        peak = peak_settings(self.cfg)
        vis_on = visibility_settings(self.cfg)["enabled"]
        # loss.visibility: the weight is the controller's when follow_visibility is set, else 1.
        vis_weight = self.vis_weight if self.visibility_follow_settings() is not None else 1.0
        visible = None
        if image_scale > 0:
            # loss.texture_mask: the L2 loss and the peak penalty see the residual weighted
            # by what each pixel's surroundings can hide (losses.texture_weights).
            residual = out["residual"]
            if texture_settings(self.cfg)["enabled"]:
                cost = texture_weights(tiles, self.cfg)
                l2_texture_loss = (cost * residual.pow(2)).mean()
                loss = loss + image_scale * weights["l2_weight"] * l2_texture_loss
                residual = residual * cost.sqrt()
            else:
                loss = loss + image_scale * weights["l2_weight"] * l2_loss
            if peak["enabled"]:
                # On the residual before the clamp, like the L2 loss, and on the same schedule:
                # the mark only piles up once the image loss is on.
                peak_loss = peak_penalty(residual, self.cfg)
                loss = loss + image_scale * float(peak["weight"]) * peak_loss
            if self.perceptual is not None:
                lpips_loss = self.perceptual(out["marked"], tiles)
                loss = loss + image_scale * weights["lpips_weight"] * lpips_loss
            if vis_on:
                visible = tile_visibility(tiles, out["marked"], self.cfg)
                loss = loss + image_scale * vis_weight * visible["mean"].mean()
        if vis_on and visible is None:
            # The image loss is not on yet: measure for the logs only.
            with torch.no_grad():
                visible = tile_visibility(tiles, out["marked"].detach(), self.cfg)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.cfg["train"]["grad_clip"]
        if clip:
            torch.nn.utils.clip_grad_norm_(self.params, clip)
        strength = self.sim_strength()
        self.optimizer.step()
        self.step += 1

        with torch.no_grad():
            metrics: dict = bit_metrics(out["logits"], bits)
            self.update_schedule(metrics["bit_acc"])
            if visible is not None:
                if self.visibility_follow_settings() is not None:
                    # The controller's reading: the quantile over its window of per-tile values, not over this batch.
                    seen = self.observe_visibility(visible["p95"].detach().tolist())
                else:
                    seen = batch_statistic(visible["p95"].detach(), self.visibility_quantile()).item()
                metrics.update(
                    visibility=seen, visibility_mean=visible["mean"].detach().mean().item(), vis_weight=vis_weight
                )
                if math.isfinite(seen):
                    self.update_visibility(seen)
            metrics.update(
                loss=loss.item(),
                bit_loss=bit_loss.item(),
                l2_loss=l2_loss.item(),
                bit_acc_ema=self.acc_ema,
                psnr=psnr(out["marked"], tiles),
                image_scale=image_scale,
                sim_strength=strength,
                concentration=concentration(out["residual"], self.cfg),
                hidden_fraction=out["hidden_fraction"],
            )
            if peak_loss is not None:
                metrics["peak_loss"] = peak_loss.item()
            if l2_texture_loss is not None:
                metrics["l2_texture_loss"] = l2_texture_loss.item()
            if lpips_loss is not None:
                metrics["lpips_loss"] = lpips_loss.item()
            if keep_outputs:
                metrics["outputs"] = {
                    "original": tiles.detach(),
                    "marked": out["marked"].detach(),
                    "view": out["view"].detach(),
                }
            return metrics

    @torch.no_grad()
    def evaluate(self, tiles: torch.Tensor, keep_outputs: bool = False, hide: bool = False) -> dict:
        """Metrics in eval mode on fresh random bits and a fresh view.

        The reader sees its whole patch unless ``hide`` is set and spread.hide is on.
        """
        self.embedder.eval()
        self.reader.eval()
        tiles = tiles.to(self.device)
        bits = self.random_bits(tiles.shape[0])
        out = self.forward(tiles, bits, hide=hide)
        metrics: dict = bit_metrics(out["logits"], bits)
        metrics["psnr"] = psnr(out["marked"], tiles)
        metrics["concentration"] = concentration(out["residual"], self.cfg)
        if visibility_settings(self.cfg)["enabled"]:
            seen = tile_visibility(tiles, out["marked"], self.cfg)["p95"]
            metrics["visibility"] = batch_statistic(seen, self.visibility_quantile()).item()
        if keep_outputs:
            metrics["outputs"] = {"original": tiles, "marked": out["marked"], "view": out["view"]}
        return metrics


# Scalars written to TensorBoard, as tag -> key in the metrics of train_step.
SCALARS = {
    "read/bit_accuracy": "bit_acc",
    "read/message_decode_rate_exact": "message_exact",
    "read/message_decode_rate_bch": "message_correctable",
    "image/psnr_db": "psnr",
    "loss/total": "loss",
    "loss/bit": "bit_loss",
    "loss/l2": "l2_loss",  # always the plain mean square, so runs with and without the texture mask compare
    "loss/l2_texture": "l2_texture_loss",  # the L2 term the loss used, when loss.texture_mask is on
    "loss/lpips": "lpips_loss",
    "loss/peak": "peak_loss",
    "image/concentration": "concentration",
    "image/visibility": "visibility",
    "schedule/vis_weight": "vis_weight",
    "read/hidden_fraction": "hidden_fraction",
    "schedule/image_loss_scale": "image_scale",
    "schedule/sim_strength": "sim_strength",
}


def log_scalars(writer, metrics: dict, step: int) -> None:
    for tag, key in SCALARS.items():
        if key in metrics:
            writer.add_scalar(tag, metrics[key], step)


def require_gpu(device: torch.device, expect: str | None) -> None:
    """Print what the GPU is; stop if there is none or it is not the expected one.

    ``expect`` is a part of the device name, for example "RTX 4070". With it
    set the run has to be on that GPU: a CPU or MPS device stops the run too.
    """
    if device.type != "cuda":
        if expect:
            raise SystemExit(
                f"stopped: train.expect_gpu is {expect!r} but the device is {device}. "
                f"torch.cuda.is_available() is {torch.cuda.is_available()}. Not falling back."
            )
        return
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        raise SystemExit(
            f"stopped: the device is {device} but this PyTorch ({torch.__version__}) cannot use CUDA. "
            "Either the build has no CUDA support or no NVIDIA driver is reachable. Not falling back to the CPU."
        )
    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    free, total = torch.cuda.mem_get_info(index)
    print(f"torch.cuda.get_device_name({index}): {name}")
    print(f"GPU memory free: {free / 2**20:.0f} MiB of {total / 2**20:.0f} MiB")
    if expect and expect.lower() not in name.lower():
        raise SystemExit(f"stopped: expected a GPU named like {expect!r}, cuda:{index} is {name!r}. Not falling back.")


def final_report(trainer: "Trainer", batches, directory: str | Path, count: int, batches_averaged: int = 8) -> None:
    """Eval-mode metrics on fresh batches and a sample sheet, at the end of a run."""
    results = [trainer.evaluate(next(batches), keep_outputs=True) for _ in range(batches_averaged)]

    def mean(key: str) -> float:
        return sum(m[key] for m in results) / len(results)

    outputs = results[0]["outputs"]
    sheet = save_image(
        sample_sheet(outputs["original"][:count], outputs["marked"][:count]),
        Path(directory) / "samples" / f"final_step_{trainer.step:08d}.png",
        footer=trainer.cfg["data"].get("credit"),
    )
    print(
        f"final, eval mode, {batches_averaged} fresh batches at simulation strength {trainer.sim_strength():.2f}: "
        f"bit_acc {mean('bit_acc'):.3f}  all 64 bits right {mean('message_exact'):.3f}  "
        f"within BCH reach {mean('message_correctable'):.3f}  psnr {mean('psnr'):.1f} dB  "
        f"strongest spot {mean('concentration'):.1f} times the tile average\n"
        f"sample sheet (original | marked | difference x10): {sheet}",
        flush=True,
    )
    if hide_settings(trainer.cfg)["enabled"]:
        hidden = [trainer.evaluate(next(batches), hide=True) for _ in range(batches_averaged)]
        acc = sum(m["bit_acc"] for m in hidden) / len(hidden)
        print(f"the same with regions of the reader's patch blanked as in training: bit_acc {acc:.3f}", flush=True)


def keep_awake() -> None:
    """Keep the machine from idle-sleeping while this process lives (Windows only).

    The request belongs to the process and ends with it; no power setting is
    changed. A closed lid still sleeps.
    """
    if os.name != "nt":
        print("--keep-awake does nothing on this system")
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    print("keeping the computer awake while this run lives (a closed lid still sleeps)")


def save_sheet(outputs: dict, step: int, count: int, directory: str | Path, credit: str | None = None) -> Path:
    """Sample sheet for this step: original | marked | difference x10, one row per sample.

    ``credit`` (data.credit) is printed below it.
    """
    original, marked = (outputs[k][:count].float().cpu() for k in ("original", "marked"))
    return save_image(sample_sheet(original, marked), Path(directory) / "samples" / f"step_{step:08d}.png", footer=credit)


def log_images(writer, outputs: dict, step: int, count: int, credit: str | None = None) -> None:
    original, marked, view = (outputs[k][:count].float().cpu() for k in ("original", "marked", "view"))
    writer.add_images("samples/1_original", original, step)
    writer.add_images("samples/2_marked", marked.clamp(0, 1), step)
    writer.add_images("samples/3_residual_x10", residual_image(marked - original), step)
    writer.add_images("samples/4_reader_input", view.clamp(0, 1), step)
    writer.add_image("samples/sheet_original_marked_residual_x10", add_footer(sample_sheet(original, marked), credit), step)


class StopRequest:
    """Turns Ctrl-C and SIGTERM into a stop at the next step boundary, with a checkpoint.

    A second signal stops at once, without one.
    """

    def __init__(self) -> None:
        self.signal_name: str | None = None
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        if self.signal_name is not None:
            raise KeyboardInterrupt
        self.signal_name = signal.Signals(signum).name
        print(f"\n{self.signal_name} received: saving a checkpoint after this step (send it again to stop now)", flush=True)


DIRECT_NOTICE = (
    "NOTE: reader_input=direct (no locator, no flatten) has not yet been shown to learn. "
    "This run is an experiment, not a known-good configuration."
)

EXIT_NOT_LEARNING = 3
EXIT_PILED_UP = 4
EXIT_IMAGE_LOSS_OFF = 5
EXIT_OVER_BUDGET = 6
EXIT_READING_LOW = 7
EXIT_INTERRUPTED = 130


def spread_summary(cfg: dict) -> str:
    """One line saying which of the two spread options are on."""
    peak, hide = peak_settings(cfg), hide_settings(cfg)
    peak_text = (
        f"on (weight {peak['weight']:g}, strongest {float(peak['top_fraction']) * 100:g} percent of "
        f"{peak['window_px']} px windows, with the image loss)"
        if peak["enabled"]
        else "off"
    )
    hide_text = (
        f"on (up to {float(hide['max_fraction']) * 100:g} percent of the reader's patch blanked, training steps only)"
        if hide["enabled"]
        else "off"
    )
    texture = texture_settings(cfg)
    texture_text = (
        f"on (L2 and peak penalty cost {float(texture['flat_weight']):g}x on flat colour, "
        f"{float(texture['busy_weight']):g}x on busy detail)"
        if texture["enabled"]
        else "off"
    )
    return f"spread: peak penalty {peak_text}; hiding {hide_text}; texture mask {texture_text}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fresh", action="store_true", help="do not resume; existing checkpoints are moved aside")
    parser.add_argument(
        "--price-per-hour",
        type=float,
        default=None,
        help="what the machine costs per hour, in dollars; adds the cost so far to every report line",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="stop after this many minutes of training in this process, with a checkpoint and a final report",
    )
    parser.add_argument("--keep-awake", action="store_true", help="keep the computer from idle-sleeping during the run")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    # Intermixed, so overrides may come before, between or after the options (pod_run.sh appends its own).
    args = parser.parse_intermixed_args()
    if args.price_per_hour is not None and args.price_per_hour < 0:
        parser.error("--price-per-hour cannot be negative")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")

    cfg = load_config(args.config, args.overrides)
    if cfg["data"].get("num_workers") is None:
        parser.error(
            f"data.num_workers is not set in {args.config}: add data.num_workers=N to the command. "
            "On a pod, setup_runpod.sh times the label generator and prints the fastest N."
        )
    check_demo(cfg)
    if int(cfg["train"]["stage"]) == 2:
        # Stage 2, the locator: its own loop (locator_train.py), same command-line options.
        from locator_train import run as run_locator

        return run_locator(cfg, args)
    ckpt_cfg, log_cfg = cfg["checkpoint"], cfg["log"]
    early = cfg.get("early_stop") or {}
    check_enabled = bool(early.get("enabled", False))
    check_after_s = float(early.get("after_minutes", 30)) * 60.0
    check_min_acc = float(early.get("min_bit_acc", 0.55))
    image_check = cfg.get("image_loss_check") or {}
    image_check_enabled = bool(image_check.get("enabled", False))
    image_check_step = int(image_check.get("at_step", 15000))
    image_check_min_psnr = float(image_check.get("min_psnr_db", 14.0))
    reading_check = cfg.get("reading_check") or {}
    reading_check_enabled = bool(reading_check.get("enabled", False))
    reading_check_step = int(reading_check.get("at_step", 80000))
    reading_check_min_acc = float(reading_check.get("min_bit_acc", 0.9))
    budget = cfg.get("budget") or {}
    max_hours, max_cost = budget.get("max_hours"), budget.get("max_cost_usd")
    if max_cost is not None and args.price_per_hour is None:
        parser.error("budget.max_cost_usd needs --price-per-hour: without a price there is no cost to limit")
    report_every = int(log_cfg.get("report_every", 1000))
    sheet_every = int(log_cfg.get("sheet_every", log_cfg["image_every"]))

    clock = RunClock(log_cfg["dir"], args.price_per_hour, fresh=args.fresh)
    if check_enabled and clock.learning_check and clock.learning_check["result"] == "failed":
        c = clock.learning_check
        print(
            f"This run was stopped by the learning check at step {c['step']}: bit accuracy {c['bit_acc']:.3f} "
            f"after {c['elapsed_minutes']:.0f} minutes, below {c['min_bit_acc']}.\n"
            "It will not resume while the check is on. To continue anyway add early_stop.enabled=false; "
            "to start over add --fresh."
        )
        return EXIT_NOT_LEARNING

    if image_check_enabled and clock.image_loss_check and clock.image_loss_check["result"] == "failed":
        c = clock.image_loss_check
        print(
            f"This run was stopped at step {c['step']}: image loss never switched on "
            f"(PSNR {c['psnr_db']:.1f} dB, below {c['min_psnr_db']:g}).\n"
            "It will not resume while the check is on. To continue anyway add image_loss_check.enabled=false; "
            "to start over add --fresh."
        )
        return EXIT_IMAGE_LOSS_OFF

    if reading_check_enabled and clock.reading_check and clock.reading_check["result"] == "failed":
        c = clock.reading_check
        print(
            f"This run was stopped at step {c['step']}: reading too weak "
            f"(mean bit accuracy {c['bit_acc']:.3f}, below {c['min_bit_acc']:g}).\n"
            "It will not resume while the check is on. To continue anyway add reading_check.enabled=false; "
            "to start over add --fresh."
        )
        return EXIT_READING_LOW

    def over_budget() -> str | None:
        """What the run has used, if that is at or past a limit in ``budget``; else None."""
        if max_hours is not None and clock.elapsed_seconds >= float(max_hours) * 3600.0:
            return f"{clock.elapsed_seconds / 3600.0:.2f} h of training, the limit is {float(max_hours):g} h (budget.max_hours)"
        if max_cost is not None and clock.cost_usd >= float(max_cost):
            return f"${clock.cost_usd:.2f} at the price given, the limit is ${float(max_cost):g} (budget.max_cost_usd)"
        return None

    spent = over_budget()
    if spent:
        print(
            f"This run has reached its limit: {spent}.\n"
            "It will not resume at this limit. To continue, raise it (budget.max_hours=..., budget.max_cost_usd=...) "
            "or switch it off (=null); to start over add --fresh."
        )
        return EXIT_OVER_BUDGET

    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])
    require_gpu(device, cfg["train"].get("expect_gpu"))
    if args.keep_awake:
        keep_awake()
    trainer = Trainer(cfg, device)
    print(
        f"device={device}  embedder={count_parameters(trainer.embedder) / 1e6:.2f}M params  "
        f"reader={count_parameters(trainer.reader) / 1e6:.2f}M params  "
        f"reader_input={cfg['train']['reader_input']}"
    )
    print(trainer.models.band.describe())
    span = camera_range(cfg)
    print(
        "camera: not set, the loss-of-resolution step is skipped"
        if span is None
        else f"camera: {span[0]:g} to {span[1]:g} px per printed tile, drawn per sample"
    )
    if cfg["train"]["reader_input"] == "direct":
        print(DIRECT_NOTICE)
    print(spread_summary(cfg))
    vis = visibility_settings(cfg)
    if vis["enabled"]:
        from visibility import pixels_per_degree
        tile_mm = float((cfg.get("demo") or {}).get("tile_mm", 50))
        floor = vis["masking_floor"]
        print(
            f"visibility loss on: S-CIELAB masked delta E, {pixels_per_degree(tile_mm, float(vis['distance_mm'])):.1f} px per degree "
            f"({tile_mm:g} mm tile at {float(vis['distance_mm']) / 10:g} cm), masking floor "
            + ("default (sqrt of busy/flat texture weights)" if floor is None else f"{float(floor):g}")
        )
        vf = trainer.visibility_follow_settings()
        if vf is not None:
            print(
                f"visibility controller: target {vf['target']:g} (the {vf['quantile'] * 100:g}th percentile of each tile's p95 masked delta E "
                f"over a window of the last {vf['window']} tiles, across steps), "
                f"ema {vf['ema']:g}, up above target, down below {vf['low']:g} x target, rate {vf['rate']:g} per step, "
                f"weight starts at {vf['init_weight']:g} within {vf['min_weight']:g} to {vf['max_weight']:g}"
            )
    follow = trainer.follow_settings()
    if follow is not None:
        print(
            f"image loss: its weight follows bit accuracy after the trigger, up at or above {follow['floor']:g}, "
            f"down below {follow['floor'] - follow['band']:g}, by 1/{cfg['schedule']['image_loss']['ramp_steps']} per step"
        )

    newest = None
    if args.fresh:
        moved = set_aside_checkpoints(ckpt_cfg["dir"], time.strftime("%Y%m%d-%H%M%S"))
        print("starting fresh" + (f"; earlier checkpoints moved to {moved}" if moved else ""))
    else:
        newest = load_latest_checkpoint(trainer, ckpt_cfg["dir"])
        if newest is not None:
            # Different random draws after a resume than in the steps already done.
            torch.manual_seed(cfg["train"]["seed"] + trainer.step)
            print(f"resumed from {newest} at step {trainer.step}")
        else:
            print(f"starting from step 0 (no checkpoint in {ckpt_cfg['dir']})")
    print(f"checkpoints: {ckpt_cfg['dir']}   logs: {log_cfg['dir']}")
    if clock.elapsed_seconds > 0:
        print(f"time on this run before now: {clock.elapsed_seconds / 3600.0:.2f} h, {clock.cost_text()}")
    if args.price_per_hour is not None and clock.unpriced_seconds > 60:
        print(
            f"note: {clock.unpriced_seconds / 3600.0:.2f} h of this run had no --price-per-hour "
            "and is not in the cost"
        )
    image_check_pending = image_check_enabled and clock.image_loss_check is None
    if image_check_pending:
        print(
            f"image-loss check: the run stops if mean PSNR is still below {image_check_min_psnr:g} dB "
            f"at step {image_check_step}"
        )
    reading_check_pending = reading_check_enabled and clock.reading_check is None
    if reading_check_pending:
        print(
            f"reading check: the run stops if mean bit accuracy is still below {reading_check_min_acc:g} "
            f"at step {reading_check_step}"
        )
    if max_hours is not None or max_cost is not None:
        limits = [f"{float(max_hours):g} h of training"] if max_hours is not None else []
        limits += [f"${float(max_cost):g} at ${args.price_per_hour:g} per hour"] if max_cost is not None else []
        print(f"budget: the run stops, with a checkpoint, at {' or '.join(limits)}, summed over restarts")
    image_loss_off = False
    reading_low = False
    check_pending = check_enabled and clock.learning_check is None
    if check_pending:
        print(
            f"learning check: the run stops if mean bit accuracy is below {check_min_acc} "
            f"once it has trained for {check_after_s / 60:.0f} minutes"
        )

    from torch.utils.tensorboard import SummaryWriter

    # After a resume, drop only what was logged past the checkpoint.
    writer = SummaryWriter(log_cfg["dir"], purge_step=trainer.step + 1 if newest is not None else 0)
    progress = ProgressLog(log_cfg["dir"], fresh=args.fresh)
    # The spread check: the same images and messages at every sample sheet.
    stop_above = (cfg.get("spread") or {}).get("stop_above")
    check_images = int(log_cfg.get("spread_images", 256))
    spread_log = None
    piled_up: dict | None = None
    if sheet_every and check_images > 0:
        # Kept on the CPU; the check moves them to the device a few at a time.
        fixed_tiles, fixed_bits = fixed_batch_for(cfg, check_images)
        fixed_busy = busy_windows(fixed_tiles, cfg)
        spread_log = SpreadLog(log_cfg["dir"], log_cfg.get("spread_markdown"), fresh=args.fresh)
        if not spread_log.has_step(trainer.step):
            start_report = spread_report(trainer.embedder, fixed_tiles, fixed_bits, cfg, fixed_busy)
            print(spread_log.write(trainer.step, start_report, None, None), flush=True)
        if stop_above is not None:
            print(
                f"spread check: the run stops if the fixed-place measure goes above {float(stop_above):g} "
                "(a patch at the same place in every tile; a mark that follows the image stays below it)"
            )
    batches = infinite(make_loader(cfg, device))
    total = int(cfg["train"]["total_steps"])
    t0, start_step = time.time(), trainer.step
    # Time spent waiting for the next batch. Every step ends with .item() calls, so the GPU is
    # idle while the loop waits here: this is time the data loader (the CPU) held the GPU up.
    # The first batch also starts the loader's processes and is counted apart.
    first_batch_s: float | None = None
    loader_wait_s = 0.0
    # (bit accuracy, PSNR) of the most recent steps, for the report line and the learning check
    recent: deque[tuple[float, float]] = deque(maxlen=max(1, report_every))
    recent_visibility: deque[float] = deque(maxlen=max(1, report_every))
    stop = StopRequest()
    not_learning = False
    out_of_time = False
    max_seconds = None if args.max_minutes is None else args.max_minutes * 60.0
    if max_seconds is not None:
        print(f"time limit: the run stops after {args.max_minutes:g} minutes of training")

    def checkpoint_now() -> None:
        path = save_checkpoint(trainer, ckpt_cfg["dir"], int(ckpt_cfg["keep"]))
        clock.save()
        memory = ""
        if device.type == "cuda":
            memory = f"  (peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)"
        print(f"saved {path}{memory}", flush=True)

    try:
        while trainer.step < total and stop.signal_name is None:
            want_images = (trainer.step + 1) % log_cfg["image_every"] == 0
            want_sheet = bool(sheet_every) and (trainer.step + 1) % sheet_every == 0
            waited = time.perf_counter()
            tiles = next(batches)
            waited = time.perf_counter() - waited
            if first_batch_s is None:
                first_batch_s = waited
            else:
                loader_wait_s += waited
            m = trainer.train_step(tiles, keep_outputs=want_images or want_sheet)
            step = trainer.step
            clock.tick()
            recent.append((m["bit_acc"], m["psnr"]))
            if "visibility" in m and math.isfinite(m["visibility"]):
                recent_visibility.append(m["visibility"])
            if step % log_cfg["scalar_every"] == 0:
                log_scalars(writer, m, step)
            if want_images:
                log_images(writer, m["outputs"], step, int(log_cfg["num_images"]), cfg["data"].get("credit"))
            if want_sheet:
                sheet = save_sheet(m["outputs"], step, int(log_cfg["num_images"]), log_cfg["dir"], cfg["data"].get("credit"))
                print(f"sample sheet: {sheet}  (strongest spot is {m['concentration']:.1f} times the tile average)", flush=True)
            if log_cfg["print_every"] and step % log_cfg["print_every"] == 0:
                rate = (step - start_step) / max(1e-9, time.time() - t0)
                print(
                    f"step {step:6d}  bit_acc {m['bit_acc']:.3f}  decoded {m['message_correctable']:.2f}  "
                    f"psnr {m['psnr']:.1f} dB  loss {m['loss']:.4f}  image_scale {m['image_scale']:.2f}  "
                    f"sim {m['sim_strength']:.2f}  spot {m['concentration']:.1f}x  "
                    + (f"seen {m['visibility']:.2f}  w {m['vis_weight']:.3g}  " if "visibility" in m else "")
                    + f"{rate:.2f} it/s",
                    flush=True,
                )
            if report_every and step % report_every == 0:
                mean_acc = sum(a for a, _ in recent) / len(recent)
                mean_psnr = sum(p for _, p in recent) / len(recent)
                mean_seen = sum(recent_visibility) / len(recent_visibility) if recent_visibility else None
                print(
                    progress.write(
                        step, mean_acc, mean_psnr, len(recent), clock, m["image_scale"], mean_seen, m.get("vis_weight")
                    ),
                    flush=True,
                )
                if clock.price_per_hour is not None:
                    writer.add_scalar("run/cost_usd", clock.cost_usd, step)
                clock.save()
            if want_sheet and spread_log is not None:
                report = spread_report(trainer.embedder, fixed_tiles, fixed_bits, cfg, fixed_busy)
                mean_acc = sum(a for a, _ in recent) / len(recent)
                mean_psnr = sum(p for _, p in recent) / len(recent)
                print(spread_log.write(step, report, mean_acc, mean_psnr), flush=True)
                writer.add_scalar("image/fixed_place", report["fixed_place"], step)
                writer.add_scalar("image/busy_share_fixed", report["busy_share"], step)
                writer.add_scalar("image/strongest_spot_fixed", report["strongest_spot"], step)
                writer.add_scalar("image/square_share_fixed", report["square_share"], step)
                if report["visibility_p90"] == report["visibility_p90"]:  # not NaN: loss.visibility is on
                    writer.add_scalar("image/visibility_fixed", report["visibility_p90"], step)
                    writer.add_scalar("image/mark_cycles", report["mark_cycles"], step)
                    writer.add_scalar("image/colour_share", report["colour_share"], step)
                if stop_above is not None and report["fixed_place"] > float(stop_above):
                    piled_up = report
                    break
            if step % ckpt_cfg["every"] == 0 or step == total:
                checkpoint_now()
            # Wait for 100 steps so that a restart just before the mark is not judged on a handful.
            if check_pending and clock.elapsed_seconds >= check_after_s and len(recent) >= min(100, recent.maxlen):
                mean_acc = sum(a for a, _ in recent) / len(recent)
                not_learning = mean_acc < check_min_acc
                clock.learning_check = {
                    "result": "failed" if not_learning else "passed",
                    "step": step,
                    "bit_acc": mean_acc,
                    "min_bit_acc": check_min_acc,
                    "elapsed_minutes": clock.elapsed_seconds / 60.0,
                }
                check_pending = False
                clock.save()
                print(
                    f"learning check at step {step}, {clock.elapsed_seconds / 60:.0f} minutes in: "
                    f"mean bit accuracy over the last {len(recent)} steps is {mean_acc:.3f} "
                    f"(chance is 0.500, required {check_min_acc}) -> "
                    + ("NOT LEARNING, stopping the run" if not_learning else "passed, the run continues"),
                    flush=True,
                )
                if not_learning:
                    break
            # One look, at the first step at or past at_step that has 100 steps behind it in this process.
            if image_check_pending and step >= image_check_step and len(recent) >= min(100, recent.maxlen):
                mean_psnr = sum(p for _, p in recent) / len(recent)
                image_loss_off = mean_psnr < image_check_min_psnr
                clock.image_loss_check = {
                    "result": "failed" if image_loss_off else "passed",
                    "step": step,
                    "psnr_db": mean_psnr,
                    "min_psnr_db": image_check_min_psnr,
                    "image_ramp_start": trainer.image_ramp_start,
                }
                image_check_pending = False
                clock.save()
                print(
                    f"image-loss check at step {step}: mean PSNR over the last {len(recent)} steps is "
                    f"{mean_psnr:.1f} dB (required {image_check_min_psnr:g}) -> "
                    + ("image loss never switched on, stopping the run" if image_loss_off else "passed, the run continues"),
                    flush=True,
                )
                if image_loss_off:
                    break
            # One look, the same way: the first step at or past at_step with a full window behind it.
            if reading_check_pending and step >= reading_check_step and len(recent) >= min(100, recent.maxlen):
                mean_acc = sum(a for a, _ in recent) / len(recent)
                reading_low = mean_acc < reading_check_min_acc
                clock.reading_check = {
                    "result": "failed" if reading_low else "passed",
                    "step": step,
                    "bit_acc": mean_acc,
                    "min_bit_acc": reading_check_min_acc,
                }
                reading_check_pending = False
                clock.save()
                print(
                    f"reading check at step {step}: mean bit accuracy over the last {len(recent)} steps is "
                    f"{mean_acc:.3f} (required {reading_check_min_acc:g}) -> "
                    + ("reading too weak, stopping the run" if reading_low else "passed, the run continues"),
                    flush=True,
                )
                if reading_low:
                    break
            spent = over_budget()
            if spent:
                break
            if max_seconds is not None and time.time() - t0 >= max_seconds:
                out_of_time = True
                break
    except Exception:
        # Ctrl-C also reaches the data-loader workers; if their exit is what
        # failed here, the stop was asked for and the checkpoint below still happens.
        if stop.signal_name is None:
            raise
    finally:
        writer.close()
        clock.save()

    loop_s = time.time() - t0
    steps_done = trainer.step - start_step
    throughput = {
        "start_step": start_step,
        "end_step": trainer.step,
        "steps": steps_done,
        "seconds": round(loop_s, 3),
        "first_batch_seconds": None if first_batch_s is None else round(first_batch_s, 3),
        "loader_wait_seconds": round(loader_wait_s, 3),
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["data"]["num_workers"]),
        "total_steps": total,
        "price_per_hour": args.price_per_hour,
        "max_hours": max_hours,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "image_loss_on_at_end": trainer.image_ramp_start is not None,
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(Path(log_cfg["dir"]) / "throughput.json", "w", encoding="utf-8") as f:
        json.dump(throughput, f, indent=2)
    if steps_done > 0:
        after_first = max(1e-9, loop_s - (first_batch_s or 0.0))
        print(
            f"data loader: the loop waited {loader_wait_s:.1f} s for batches, {100.0 * loader_wait_s / after_first:.1f} percent "
            f"of {after_first:.0f} s after the first batch (which took {first_batch_s or 0.0:.1f} s), "
            f"with {throughput['num_workers']} workers",
            flush=True,
        )

    if not_learning:
        checkpoint_now()
        print(
            f"Stopped: bit accuracy is below the required {check_min_acc}. {clock.cost_text()}.\n"
            "The machine is still running and still billed; see RUNPOD.md for how to stop it."
        )
        return EXIT_NOT_LEARNING
    if image_loss_off:
        checkpoint_now()
        ramp = trainer.image_ramp_start
        print(
            f"Stopped: image loss never switched on. At step {trainer.step} the PSNR is still "
            f"{clock.image_loss_check['psnr_db']:.1f} dB, below {image_check_min_psnr:g} (image_loss_check). "
            + (
                f"Smoothed bit accuracy is {trainer.acc_ema:.3f} and has not passed "
                f"{cfg['schedule']['image_loss']['acc_threshold']}, which is what switches the image loss on."
                if ramp is None
                else f"Strictly, its ramp began at step {ramp}, too late to lift the PSNR by now."
            )
            + f" {clock.cost_text()}.\n"
            "The machine is still running and still billed; see RUNPOD.md for how to stop it."
        )
        return EXIT_IMAGE_LOSS_OFF
    if reading_low:
        checkpoint_now()
        print(
            f"Stopped: reading too weak. At step {trainer.step} the mean bit accuracy is "
            f"{clock.reading_check['bit_acc']:.3f}, below {reading_check_min_acc:g} (reading_check): "
            f"the mark was made invisible at the cost of reading. {clock.cost_text()}.\n"
            "The machine is still running and still billed; see RUNPOD.md for how to stop it."
        )
        return EXIT_READING_LOW
    if piled_up is not None:
        checkpoint_now()
        print(
            f"Stopped: at step {trainer.step} the fixed-place measure is {piled_up['fixed_place']:.2f}, "
            f"above the limit of {float(stop_above):g} (spread.stop_above): the mark is piling up at the same "
            f"place in every tile. Strongest spot {piled_up['strongest_spot']:.2f}, "
            f"{piled_up['square_share'] * 100:.1f} percent of the mark's energy in one 96 px square. "
            "The same command would resume and stop again at the next sample sheet unless the limit is changed."
        )
        return EXIT_PILED_UP
    if spent:
        checkpoint_now()
        print(
            f"Stopped: the run reached its limit at step {trainer.step}: {spent}. {clock.cost_text()}.\n"
            "The same command will not resume at this limit; raise it or set it to null to continue.\n"
            "The machine is still running and still billed; see RUNPOD.md for how to stop it."
        )
        return EXIT_OVER_BUDGET
    if stop.signal_name is not None:
        checkpoint_now()
        print(f"stopped by {stop.signal_name} at step {trainer.step}; the same command resumes from here")
        return EXIT_INTERRUPTED
    rate = (trainer.step - start_step) / max(1e-9, time.time() - t0)
    if out_of_time:
        checkpoint_now()
        print(
            f"time limit reached: stopped at step {trainer.step} after {args.max_minutes:g} minutes, "
            f"{rate:.2f} steps per second over this process; the same command without --max-minutes resumes from here"
        )
    else:
        print(
            f"finished {trainer.step} steps in {clock.elapsed_seconds / 3600.0:.2f} h, {clock.cost_text()}, "
            f"{rate:.2f} steps per second over this process"
        )
    final_report(trainer, batches, log_cfg["dir"], int(log_cfg["num_images"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
