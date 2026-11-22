"""Dataset: any folder of images, searched recursively."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageOps
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


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image upright, as RGB.

    A phone stores a photo as the sensor saw it and writes how to turn it in
    a tag (EXIF orientation). The tag is applied here, so a photo taken with
    the phone on its side does not reach the models on its side.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image folder not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files


HOLDOUT_SEED = 1


def holdout_images(root: str | Path, count: int) -> list[Path]:
    """The ``count`` images of a folder that training never sees (data.holdout), sorted.

    Picked by a seeded shuffle of the sorted file list, so they are the same
    files every time as long as the folder holds the same files.
    acceptance.py measures on them.
    """
    files = list_images(root)
    if not 0 <= count < len(files):
        raise ValueError(f"data.holdout must be at least 0 and below the number of images ({len(files)}), got {count}")
    picks = torch.randperm(len(files), generator=torch.Generator().manual_seed(HOLDOUT_SEED))[:count].tolist()
    return sorted(files[index] for index in picks)


class CenterFraction:
    """Cut an image down to its middle: ``fraction`` of the width and of the height.

    0.6 keeps the central 60 percent of each side, which is 36 percent of the
    area. A class, not a lambda, so that loader workers can pickle it.
    """

    def __init__(self, fraction: float):
        self.fraction = fraction

    def box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the kept part."""
        w = max(1, round(width * self.fraction))
        h = max(1, round(height * self.fraction))
        left, top = (width - w) // 2, (height - h) // 2
        return left, top, left + w, top + h

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.crop(self.box(*image.size))


class ImageFolderDataset(Dataset):
    """Returns float RGB tiles in [0,1] of shape (3, image_size, image_size).

    ``center_fraction`` below 1 makes the random crop draw from the middle of
    each image only (see CenterFraction): for photos of an object with table,
    hand or shelf around it. 1.0 crops from the whole image.

    ``holdout`` images are left out (see holdout_images); 0 uses every image.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = TILE_PX,
        min_crop_scale: float = 0.35,
        center_fraction: float = 1.0,
        holdout: int = 0,
    ):
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError(f"data.center_fraction must be above 0 and at most 1, got {center_fraction}")
        self.files = list_images(root)
        if holdout:
            held = set(holdout_images(root, holdout))
            self.files = [path for path in self.files if path not in held]
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(min_crop_scale, 1.0), ratio=(0.8, 1.25), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if center_fraction < 1.0:
            steps.insert(0, CenterFraction(center_fraction))
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.files)

    def load(self, index: int) -> Image.Image:
        """The whole image, upright, before any crop."""
        return load_rgb(self.files[index])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.load(index))


def make_loader(cfg: dict, device: torch.device) -> DataLoader:
    data_cfg = cfg["data"]
    dataset = ImageFolderDataset(
        data_cfg["train_dir"],
        image_size=TILE_PX,
        min_crop_scale=data_cfg["min_crop_scale"],
        center_fraction=float(data_cfg.get("center_fraction", 1.0)),
        holdout=int(data_cfg.get("holdout", 0)),
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


def crops_preview(cfg: dict, count: int = 64, seed: int = 0, columns: int = 8, gap: int = 4) -> tuple[torch.Tensor, list[dict]]:
    """A sheet of training crops, to look at what the models will be shown.

    ``count`` images picked by a seeded shuffle, one crop each, made by the
    dataset the training loader uses with this config's data settings.
    Returns the sheet (3,H,W) and one row per crop, in reading order: the
    file and the size in source pixels of the part that became the tile.
    """
    data_cfg = cfg["data"]
    fraction = float(data_cfg.get("center_fraction", 1.0))
    dataset = ImageFolderDataset(
        data_cfg["train_dir"], TILE_PX, data_cfg["min_crop_scale"], fraction, int(data_cfg.get("holdout", 0))
    )
    picks = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    tiles, rows = [], []
    for position, index in enumerate(picks):
        crop_seed = seed * 1_000_003 + position
        whole = dataset.load(index)
        torch.manual_seed(crop_seed)
        tiles.append(dataset.transform(whole))
        # The crop's place is the first thing the transform draws, so the same
        # seed gives it again.
        torch.manual_seed(crop_seed)
        width, height = whole.size
        region = CenterFraction(fraction).box(width, height)
        blank = Image.new("L", (region[2] - region[0], region[3] - region[1]))
        _, _, h, w = transforms.RandomResizedCrop.get_params(blank, [data_cfg["min_crop_scale"], 1.0], [0.8, 1.25])
        rows.append(
            {
                "row": position // columns + 1,
                "column": position % columns + 1,
                "file": dataset.files[index].name,
                "image_px": f"{width}x{height}",
                "crop_px": f"{w}x{h}",
                "enlarged": round(TILE_PX / min(w, h), 2),  # above 1: the crop was scaled up to the tile
            }
        )
    lines = -(-len(tiles) // columns)
    sheet = torch.ones(3, lines * TILE_PX + (lines - 1) * gap, columns * TILE_PX + (columns - 1) * gap)
    for position, tile in enumerate(tiles):
        top, left = (position // columns) * (TILE_PX + gap), (position % columns) * (TILE_PX + gap)
        sheet[:, top : top + TILE_PX, left : left + TILE_PX] = tile
    return sheet, rows


if __name__ == "__main__":
    import argparse
    import csv

    from config import load_config
    from viz import save_image

    parser = argparse.ArgumentParser(description="Save a sheet of training crops made with a config's data settings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sheet", required=True, help="PNG to write; a .csv with the same name lists the crops")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    preview, listing = crops_preview(config, args.count, args.seed)
    target = save_image(preview, args.sheet, footer=config["data"].get("credit"))
    with open(target.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(listing[0]))
        writer.writeheader()
        writer.writerows(listing)
    enlarged = sum(row["enlarged"] > 1.0 for row in listing)
    print(
        f"{len(listing)} crops from {config['data']['train_dir']}, center_fraction "
        f"{config['data'].get('center_fraction', 1.0)}, min_crop_scale {config['data']['min_crop_scale']}"
    )
    print(f"{enlarged} of {len(listing)} crops were scaled up to {TILE_PX} px, the largest by {max(row['enlarged'] for row in listing)}x")
    print(f"sheet: {target}\nlist:  {target.with_suffix('.csv')}")
