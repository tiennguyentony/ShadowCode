"""Stage 1 training: embedder and reader.

    python train.py --config configs/default.yaml data.train_dir=/path/to/images

Resumes automatically from the newest checkpoint in ``checkpoint.dir``.
Pass --fresh to start over; existing checkpoints are moved aside, not deleted.

Every ``log.report_every`` steps one line goes to the console and to
``<log.dir>/progress.csv``: step, bit accuracy, PSNR and, with
``--price-per-hour``, the cost so far.

    python train.py --config configs/runpod_direct.yaml --price-per-hour 0.69

Ctrl-C or SIGTERM saves a checkpoint and exits. Exit codes: 0 finished or
stopped by ``--max-minutes``, 3 stopped by the learning check
(``early_stop``), 4 stopped because the mark piled up at a fixed place in the
tile (``spread.stop_above``),
5 stopped because the image loss never switched on (``image_loss_check``),
130 interrupted.

With every sample sheet the run measures where the mark sits on 256 fixed
dataset images (the spread check) and appends a row to ``<log.dir>/spread.csv``.

On a CUDA device the run first prints what the GPU is and how much memory is
free. ``train.expect_gpu`` names the GPU the run must be on; anything else,
including no CUDA at all, stops the run. It never falls back to the CPU then.

    python train.py --config configs/sanity.yaml --max-minutes 60
"""

from __future__ import annotations

import argparse
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
from models import build_models, count_parameters
from runlog import ProgressLog, RunClock, SpreadLog
from simulate import simulate, wrap_surface
from spread import (
    busy_windows,
    concentration,
    fixed_batch,
    hide_regions,
    hide_settings,
    peak_penalty,
    peak_settings,
    spread_report,
)
from viz import residual_image, sample_sheet, save_image

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
      ``ramp_steps`` steps.
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

    # -- schedules ---------------------------------------------------------

    def image_loss_scale(self) -> float:
        if self.image_ramp_start is None:
            return 0.0
        steps = int(self.cfg["schedule"]["image_loss"]["ramp_steps"])
        if steps <= 0:
            return 1.0
        return min(1.0, (self.step - self.image_ramp_start) / steps)

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
        peak = peak_settings(self.cfg)
        if image_scale > 0:
            loss = loss + image_scale * weights["l2_weight"] * l2_loss
            if peak["enabled"]:
                # On the residual before the clamp, like the L2 loss, and on the same schedule:
                # the mark only piles up once the image loss is on.
                peak_loss = peak_penalty(out["residual"], self.cfg)
                loss = loss + image_scale * float(peak["weight"]) * peak_loss
            if self.perceptual is not None:
                lpips_loss = self.perceptual(out["marked"], tiles)
                loss = loss + image_scale * weights["lpips_weight"] * lpips_loss

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
            sched = self.cfg["schedule"]["image_loss"]
            self.acc_ema = sched["acc_ema"] * self.acc_ema + (1.0 - sched["acc_ema"]) * metrics["bit_acc"]
            if self.image_ramp_start is None and self.acc_ema > sched["acc_threshold"]:
                self.image_ramp_start = self.step
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
    "loss/l2": "l2_loss",
    "loss/lpips": "lpips_loss",
    "loss/peak": "peak_loss",
    "image/concentration": "concentration",
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


def save_sheet(outputs: dict, step: int, count: int, directory: str | Path) -> Path:
    """Sample sheet for this step: original | marked | difference x10, one row per sample."""
    original, marked = (outputs[k][:count].float().cpu() for k in ("original", "marked"))
    return save_image(sample_sheet(original, marked), Path(directory) / "samples" / f"step_{step:08d}.png")


def log_images(writer, outputs: dict, step: int, count: int) -> None:
    original, marked, view = (outputs[k][:count].float().cpu() for k in ("original", "marked", "view"))
    writer.add_images("samples/1_original", original, step)
    writer.add_images("samples/2_marked", marked.clamp(0, 1), step)
    writer.add_images("samples/3_residual_x10", residual_image(marked - original), step)
    writer.add_images("samples/4_reader_input", view.clamp(0, 1), step)
    writer.add_image("samples/sheet_original_marked_residual_x10", sample_sheet(original, marked), step)


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
    return f"spread: peak penalty {peak_text}; hiding {hide_text}"


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
    args = parser.parse_args()
    if args.price_per_hour is not None and args.price_per_hour < 0:
        parser.error("--price-per-hour cannot be negative")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")

    cfg = load_config(args.config, args.overrides)
    check_demo(cfg)
    ckpt_cfg, log_cfg = cfg["checkpoint"], cfg["log"]
    early = cfg.get("early_stop") or {}
    check_enabled = bool(early.get("enabled", False))
    check_after_s = float(early.get("after_minutes", 30)) * 60.0
    check_min_acc = float(early.get("min_bit_acc", 0.55))
    image_check = cfg.get("image_loss_check") or {}
    image_check_enabled = bool(image_check.get("enabled", False))
    image_check_step = int(image_check.get("at_step", 15000))
    image_check_min_psnr = float(image_check.get("min_psnr_db", 14.0))
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
    image_loss_off = False
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
        fixed_tiles, fixed_bits = fixed_batch(cfg["data"]["train_dir"], check_images)
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
    # (bit accuracy, PSNR) of the most recent steps, for the report line and the learning check
    recent: deque[tuple[float, float]] = deque(maxlen=max(1, report_every))
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
            m = trainer.train_step(next(batches), keep_outputs=want_images or want_sheet)
            step = trainer.step
            clock.tick()
            recent.append((m["bit_acc"], m["psnr"]))
            if step % log_cfg["scalar_every"] == 0:
                log_scalars(writer, m, step)
            if want_images:
                log_images(writer, m["outputs"], step, int(log_cfg["num_images"]))
            if want_sheet:
                sheet = save_sheet(m["outputs"], step, int(log_cfg["num_images"]), log_cfg["dir"])
                print(f"sample sheet: {sheet}  (strongest spot is {m['concentration']:.1f} times the tile average)", flush=True)
            if log_cfg["print_every"] and step % log_cfg["print_every"] == 0:
                rate = (step - start_step) / max(1e-9, time.time() - t0)
                print(
                    f"step {step:6d}  bit_acc {m['bit_acc']:.3f}  decoded {m['message_correctable']:.2f}  "
                    f"psnr {m['psnr']:.1f} dB  loss {m['loss']:.4f}  image_scale {m['image_scale']:.2f}  "
                    f"sim {m['sim_strength']:.2f}  spot {m['concentration']:.1f}x  {rate:.2f} it/s",
                    flush=True,
                )
            if report_every and step % report_every == 0:
                mean_acc = sum(a for a, _ in recent) / len(recent)
                mean_psnr = sum(p for _, p in recent) / len(recent)
                print(progress.write(step, mean_acc, mean_psnr, len(recent), clock), flush=True)
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
