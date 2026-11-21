"""Dataset: any folder of images, searched recursively."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from constants import TILE_PX

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def procedural_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """One generated image (smooth colour field, shapes, grain) as (size, size, 3) uint8.

    Used where real images are not wanted: the smoke test's dataset and the
    fixtures of an exported bundle. Nothing is downloaded and no licence applies.
    """
    low = rng.random((rng.integers(3, 9), rng.integers(3, 9), 3)).astype(np.float32)
    img = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
        dtype=np.float32,
    )
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(rng.integers(2, 7)):
        cx, cy, r = rng.integers(0, size), rng.integers(0, size), rng.integers(16, 110)
        colour = rng.random(3) * 255
        if rng.random() < 0.5:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        else:
            mask = (abs(xx - cx) < r) & (abs(yy - cy) < r // 2 + 4)
        img[mask] = colour
    img += rng.normal(0, rng.uniform(0, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image folder not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files


class ImageFolderDataset(Dataset):
    """Returns float RGB tiles in [0,1] of shape (3, image_size, image_size)."""

    def __init__(self, root: str | Path, image_size: int = TILE_PX, min_crop_scale: float = 0.35):
        self.files = list_images(root)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(min_crop_scale, 1.0), ratio=(0.8, 1.25), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as img:
            return self.transform(img.convert("RGB"))


def make_loader(cfg: dict, device: torch.device) -> DataLoader:
    data_cfg = cfg["data"]
    dataset = ImageFolderDataset(
        data_cfg["train_dir"],
        image_size=TILE_PX,
        min_crop_scale=data_cfg["min_crop_scale"],
    )
    workers = int(data_cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=len(dataset) >= cfg["train"]["batch_size"],
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader
