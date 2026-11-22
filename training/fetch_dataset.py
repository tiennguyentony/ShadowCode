"""Download the training images: Open Images V7, validation split.

    python fetch_dataset.py --out /workspace/datasets/open-images-v7-val
    python fetch_dataset.py --out data/open-images-v7-val --count 2000
    python fetch_dataset.py --out ... --count all

Images are listed as CC BY 2.0; see THIRD_PARTY.md for the licence position.
The default is the first 16,000 images by image ID, 4.94 GB. The whole split
is 41,620 images, 12.88 GB (both figures summed from the bucket listing on
2026-11-21). Anything over 5 GB needs --allow-large.

Writes into --out:

    images/<ImageID>.jpg   point data.train_dir here
    ATTRIBUTION.csv        author, title, licence and source URL of every image fetched
    image_list.csv         the split's full list as published (15 MB)
    COMPLETE               written last; holds the number of images

Safe to run again: files already there are skipped, and an unfinished file is
never left under its final name. Uses the standard library only.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NAME = "Open Images V7, validation split"
LIST_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
EXPECTED_LICENCE = "https://creativecommons.org/licenses/by/2.0/"
SPLIT_IMAGES = 41_620
DEFAULT_COUNT = 16_000
# Mean size of one image in the split (12.88 GB / 41,620). Only used for the
# estimate printed before a download; the default subset's own figure is exact.
MEAN_IMAGE_BYTES = 309_500
DEFAULT_SUBSET_BYTES = 4_941_000_000
LARGE_BYTES = 5_000_000_000
ATTRIBUTION_COLUMNS = ["ImageID", "Author", "AuthorProfileURL", "Title", "License", "OriginalURL", "OriginalLandingURL"]


def fetch(url: str, target: Path, attempts: int = 4) -> int:
    """Download url to target through a temporary name. Returns the bytes written."""
    temporary = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
                expected = response.headers.get("Content-Length")
                written = 0
                while chunk := response.read(1 << 16):
                    out.write(chunk)
                    written += len(chunk)
            if expected is not None and written != int(expected):
                raise OSError(f"got {written} of {expected} bytes")
            os.replace(temporary, target)
            return written
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == attempts:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


def estimated_bytes(count: int) -> int:
    return DEFAULT_SUBSET_BYTES if count == DEFAULT_COUNT else count * MEAN_IMAGE_BYTES


def is_complete(out: Path, count: int) -> bool:
    marker = out / "COMPLETE"
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == str(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="dataset folder; images go to <out>/images")
    parser.add_argument("--count", default=str(DEFAULT_COUNT), help=f"images to fetch, or 'all' (default {DEFAULT_COUNT})")
    parser.add_argument("--workers", type=int, default=32, help="parallel downloads (default 32)")
    parser.add_argument("--allow-large", action="store_true", help="permit a download over 5 GB")
    parser.add_argument("--check", action="store_true", help="only report whether the dataset is already complete")
    args = parser.parse_args()

    count = SPLIT_IMAGES if args.count == "all" else int(args.count)
    if not 0 < count <= SPLIT_IMAGES:
        parser.error(f"--count must be between 1 and {SPLIT_IMAGES}, or 'all'")
    out = Path(args.out)
    images = out / "images"
    if is_complete(out, count):
        print(f"dataset already complete: {count} images in {images}")
        return 0
    if args.check:
        print(f"dataset not complete in {out}")
        return 1

    size = estimated_bytes(count)
    print(f"dataset:  {NAME}, first {count} of {SPLIT_IMAGES} images by image ID")
    print(f"licence:  images listed as CC BY 2.0 ({EXPECTED_LICENCE}); see THIRD_PARTY.md")
    print(f"size:     about {size / 1e9:.2f} GB into {out}")
    print("time:     not measured on a pod; the rate is printed below as it runs")
    if size > LARGE_BYTES and not args.allow_large:
        print("\nThis is over 5 GB. Nothing was downloaded. Run again with --allow-large to go ahead.")
        return 2

    images.mkdir(parents=True, exist_ok=True)
    list_path = out / "image_list.csv"
    if not list_path.is_file():
        print("fetching the image list (15 MB)")
        fetch(LIST_URL, list_path)
    with open(list_path, "r", encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: row["ImageID"])
    if len(rows) != SPLIT_IMAGES:
        print(f"the image list has {len(rows)} rows, expected {SPLIT_IMAGES}; stopping")
        return 1
    rows = rows[:count]
    other = sorted({row["License"] for row in rows} - {EXPECTED_LICENCE})
    if other:
        print(f"the image list names licences other than CC BY 2.0: {other}; stopping")
        return 1
    for row in rows:
        if not row["ImageID"].isalnum():
            print(f"unexpected image ID {row['ImageID']!r}; stopping")
            return 1

    todo = [row["ImageID"] for row in rows if not (images / f"{row['ImageID']}.jpg").is_file()]
    print(f"{count - len(todo)} images already present, {len(todo)} to download with {args.workers} workers")
    failures: list[str] = []
    done = downloaded = 0
    started = last_print = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(fetch, IMAGE_URL.format(image_id=image_id), images / f"{image_id}.jpg"): image_id
            for image_id in todo
        }
        for job in as_completed(jobs):
            done += 1
            try:
                downloaded += job.result()
            except RuntimeError as error:
                failures.append(str(error))
            now = time.time()
            if now - last_print > 10 or done == len(todo):
                last_print = now
                rate = downloaded / max(1e-9, now - started)
                remaining = (len(todo) - done) * MEAN_IMAGE_BYTES / max(1.0, rate)
                print(
                    f"  {done}/{len(todo)}  {downloaded / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s  "
                    f"about {remaining / 60:.0f} min left",
                    flush=True,
                )

    if failures:
        print(f"\n{len(failures)} downloads failed, for example:\n  " + "\n  ".join(failures[:5]))
        print("Run the same command again to retry only the missing images.")
        return 1

    with open(out / "ATTRIBUTION.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTRIBUTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "COMPLETE").write_text(f"{count}\n", encoding="utf-8")
    print(f"done: {count} images in {images} ({(time.time() - started) / 60:.1f} min)")
    print(f"attribution for every image: {out / 'ATTRIBUTION.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
