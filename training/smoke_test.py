"""Smoke test: 300 training steps on a tiny generated dataset, distortions off.

Asserts that bit accuracy rises well above chance (50 percent). Runs on CPU,
or MPS on Apple silicon. Uses the real models and the real training step:
encode, tile 3x3, random crop at an arbitrary tile offset, decode.

    python smoke_test.py
    pytest smoke_test.py -s
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import load_config
from data import infinite, make_loader
from models import count_parameters
from train import Trainer

HERE = Path(__file__).resolve().parent
REQUIRED_ACCURACY = 0.70  # chance is 0.50
NUM_IMAGES = 24


def make_tiny_dataset(root: Path, count: int = NUM_IMAGES, size: int = 320, seed: int = 0) -> None:
    """Write procedural images (smooth colour fields, shapes, grain) to root."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        low = rng.random((rng.integers(3, 9), rng.integers(3, 9), 3)).astype(np.float32)
        img = np.asarray(
            Image.fromarray((low * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
            dtype=np.float32,
        )
        yy, xx = np.mgrid[0:size, 0:size]
        for _ in range(rng.integers(2, 7)):
            cx, cy, r = rng.integers(0, size), rng.integers(0, size), rng.integers(12, 80)
            colour = rng.random(3) * 255
            if rng.random() < 0.5:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
            else:
                mask = (abs(xx - cx) < r) & (abs(yy - cy) < r // 2 + 4)
            img[mask] = colour
        img += rng.normal(0, rng.uniform(0, 12), img.shape)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / f"img_{i:03d}.png")


def default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_smoke(device_name: str | None = None) -> dict[str, float]:
    device = torch.device(device_name or default_device())
    with tempfile.TemporaryDirectory(prefix="tilemark_smoke_") as tmp:
        make_tiny_dataset(Path(tmp))
        cfg = load_config(HERE / "configs" / "smoke.yaml", [f"data.train_dir={Path(tmp).as_posix()}"])
        torch.manual_seed(cfg["train"]["seed"])
        trainer = Trainer(cfg, device)
        steps = cfg["train"]["total_steps"]
        every = cfg["log"]["print_every"]
        print(
            f"smoke test: {steps} steps, batch {cfg['train']['batch_size']}, {NUM_IMAGES} images, "
            f"device={device}, distortions off"
        )
        print(
            f"encoder {count_parameters(trainer.encoder) / 1e6:.2f}M params, "
            f"decoder {count_parameters(trainer.decoder) / 1e6:.2f}M params"
        )

        batches = infinite(make_loader(cfg, device))
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
                    f"{time.time() - t0:5.0f}s"
                )

        first = sum(m["bit_acc"] for m in history[:every]) / every
        final = sum(m["bit_acc"] for m in history[-every:]) / every
        eval_accs = [trainer.evaluate(next(batches)) for _ in range(8)]
        eval_acc = sum(eval_accs) / len(eval_accs)

    print(f"\nfirst {every} steps:  bit_acc {first:.3f}")
    print(f"last {every} steps:   bit_acc {final:.3f}")
    print(f"eval mode, fresh messages and offsets: bit_acc {eval_acc:.3f}")
    print(f"total time: {time.time() - t0:.0f}s")
    return {"first": first, "final": final, "eval": eval_acc}


def test_smoke() -> None:
    result = run_smoke()
    assert result["final"] >= REQUIRED_ACCURACY, (
        f"training bit accuracy {result['final']:.3f} did not reach {REQUIRED_ACCURACY}"
    )
    assert result["eval"] >= REQUIRED_ACCURACY, (
        f"eval bit accuracy {result['eval']:.3f} did not reach {REQUIRED_ACCURACY}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="cpu or mps (default: mps if available, else cpu)")
    args = parser.parse_args()
    result = run_smoke(args.device)
    assert result["final"] >= REQUIRED_ACCURACY, (
        f"FAIL: training bit accuracy {result['final']:.3f} < {REQUIRED_ACCURACY}"
    )
    assert result["eval"] >= REQUIRED_ACCURACY, f"FAIL: eval bit accuracy {result['eval']:.3f} < {REQUIRED_ACCURACY}"
    print(f"PASS: bit accuracy {result['final']:.3f} (train) / {result['eval']:.3f} (eval) >= {REQUIRED_ACCURACY}")
