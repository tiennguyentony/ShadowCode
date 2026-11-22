"""Download the label images: Open Food Facts front-of-pack photos.

    python fetch_labels.py --out data/open-food-facts
    python fetch_labels.py --out data/open-food-facts --count 2000
    python fetch_labels.py --out data/open-food-facts --list-only
    python fetch_labels.py --out data/open-food-facts --check

Images are CC BY-SA 3.0; see THIRD_PARTY.md for the licence position and the
open points (label artwork belongs to third parties; share-alike and weights).

Which images. One per product: the upload that the product's front-of-pack
selection points at, and only when the selection is that whole upload (same
pixel size), because the bulk bucket holds uploads, not the cropped
selections. Shorter side at least --min-side pixels (default 600). Products
are drawn at random, with a fixed seed, from across the whole database.

Where from, following Open Food Facts' own instructions for bulk use:

    product list   the Parquet export on Hugging Face; only the footer and the
                   columns code, lang and images of the sampled row groups are
                   read, with HTTP range requests (about 0.2 GB, not 7.7 GB)
    list of files  data_keys.gz in the images bucket (66 MB)
    images         the openfoodfacts-images bucket on AWS, which Open Food
                   Facts names as the place for many images and which allows
                   parallel downloads. Nothing is fetched from the Open Food
                   Facts servers themselves and the API is not used.

Writes into --out:

    images/<barcode>_<image id>.jpg   point data.train_dir here
    ATTRIBUTION.csv    product page, uploader, licence and source URL of every image kept
    candidates.csv     the sampled products, in the order they are tried
    skipped.csv        candidates that could not be used, and why
    source.json        which export the list came from, the seed and the date
    data_keys.gz       the bucket's file list as published
    COMPLETE           written last; holds the number of images

Safe to run again: files already there are skipped, an unfinished file is
never left under its final name, and the product list is only sampled once.
Needs pyarrow (for the product list) and Pillow (to check each image).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NAME = "Open Food Facts, front-of-pack product photos"
PARQUET_URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
BUCKET_URL = "https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/"
KEYS_URL = BUCKET_URL + "data/data_keys.gz"
PRODUCT_PAGE = "https://world.openfoodfacts.org/product/{code}"
LICENCE = "https://creativecommons.org/licenses/by-sa/3.0/"
USER_AGENT = "fetch_labels.py (one-off dataset download)"
DEFAULT_COUNT = 16_000
DEFAULT_MIN_SIDE = 600
# Candidates sampled for every image wanted: some fail the check after download.
OVERSAMPLE = 1.25
# At most this many products from one row group (about 1,000 neighbouring
# barcodes, often one manufacturer), so that no brand dominates.
PER_GROUP = 40
CANDIDATE_COLUMNS = ["File", "Barcode", "ImageID", "Width", "Height", "Uploader", "UploadedUTC", "Key"]
ATTRIBUTION_COLUMNS = [
    "File", "Barcode", "ImageID", "Uploader", "UploadedUTC", "ProductPage", "ImageURL", "License", "Credit",
    "Width", "Height", "Bytes",
]


class Unusable(Exception):
    """This candidate will never work (missing from the bucket, wrong size, not an image)."""


def open_url(url: str, headers: dict | None = None, method: str = "GET"):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=120)


def fetch(url: str, target: Path, attempts: int = 4) -> int:
    """Download url to target through a temporary name. Returns the bytes written."""
    temporary = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with open_url(url) as response, open(temporary, "wb") as out:
                expected = response.headers.get("Content-Length")
                written = 0
                while chunk := response.read(1 << 16):
                    out.write(chunk)
                    written += len(chunk)
            if expected is not None and written != int(expected):
                raise OSError(f"got {written} of {expected} bytes")
            os.replace(temporary, target)
            return written
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code in (403, 404):
                raise Unusable(f"HTTP {error.code}") from error
            if attempt == attempts:
                raise RuntimeError(f"{url}: {error}") from error
            # 503 SlowDown and the like: back off, do not push.
            time.sleep(5.0 * attempt)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == attempts:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, enough of a file for pyarrow."""

    def __init__(self, url: str, size: int | None = None):
        self.url, self.position, self.fetched = url, 0, 0
        if size is None:
            with open_url(url, method="HEAD") as response:
                self.url = response.url  # after redirects, so later requests go straight there
                size = int(response.headers["Content-Length"])
                self.last_modified = response.headers.get("Last-Modified", "")
                self.etag = response.headers.get("ETag", "")
        self.size = size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.position
        if n <= 0:
            return b""
        last = min(self.size, self.position + n) - 1
        for attempt in range(1, 5):
            try:
                with open_url(self.url, {"Range": f"bytes={self.position}-{last}"}) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, OSError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(3.0 * attempt)
        self.position += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def bucket_key(code: str, image_id: int) -> str:
    """Path of an upload in the bucket: barcode padded to 13 digits, split 3/3/3/rest."""
    code = code.zfill(13)
    return f"data/{code[0:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}/{image_id}.jpg"


def front_upload(product: dict, min_side: int) -> dict | None:
    """The candidate row for one product, or None if it has no usable front photo."""
    code = product["code"] or ""
    if not (code.isdigit() and 8 <= len(code) <= 24):
        return None
    images = product["images"] or []
    fronts = {image["key"]: image for image in images if (image["key"] or "").startswith("front_")}
    if not fronts:
        return None
    # The front photo in the product's main language, else the first by name.
    front = fronts.get(f"front_{product['lang']}") or fronts[sorted(fronts)[0]]
    if front["imgid"] is None:
        return None
    upload = next((image for image in images if image["key"] == str(front["imgid"])), None)
    if upload is None:
        return None
    selected, full = (front["sizes"] or {}).get("full"), (upload["sizes"] or {}).get("full")
    if not selected or not full or not full["w"] or not full["h"]:
        return None
    if (selected["w"], selected["h"]) != (full["w"], full["h"]):
        return None  # the selection is a crop or a rotation of the upload
    if min(full["w"], full["h"]) < min_side:
        return None
    uploaded = upload["uploaded_t"]
    return {
        "File": f"{code}_{front['imgid']}.jpg",
        "Barcode": code,
        "ImageID": front["imgid"],
        "Width": full["w"],
        "Height": full["h"],
        "Uploader": upload["uploader"] or "",
        "UploadedUTC": datetime.datetime.fromtimestamp(uploaded, datetime.timezone.utc).strftime("%Y-%m-%d") if uploaded else "",
        "Key": bucket_key(code, front["imgid"]),
    }


def keys_in_bucket(out: Path, wanted: set[str]) -> set[str]:
    """Which of the wanted keys the bucket's published file list contains."""
    keys_path = out / "data_keys.gz"
    if not keys_path.is_file():
        print("fetching the bucket's file list (66 MB)", flush=True)
        fetch(KEYS_URL, keys_path)
    found: set[str] = set()
    with gzip.open(keys_path, "rt", encoding="utf-8") as lines:
        for line in lines:
            key = line.rstrip("\n")
            if key in wanted:
                found.add(key)
    return found


def sample_candidates(out: Path, wanted: int, min_side: int, seed: int, workers: int = 4) -> list[dict]:
    """Draw products at random from the Parquet export and keep those with a usable front photo."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("the product list needs pyarrow: pip install pyarrow")

    print("reading the product list's index from Hugging Face (about 0.1 GB)", flush=True)
    remote = RangeFile(PARQUET_URL)
    metadata = pq.ParquetFile(remote).metadata
    fetched = remote.fetched
    print(f"  {metadata.num_rows} products in {metadata.num_row_groups} row groups, export of {remote.last_modified}")

    rng = random.Random(seed)
    order = list(range(metadata.num_row_groups))
    rng.shuffle(order)

    def read_group(group: int) -> tuple[int, list[dict], int]:
        handle = RangeFile(remote.url, remote.size)
        table = pq.ParquetFile(handle, metadata=metadata).read_row_group(group, columns=["code", "lang", "images"])
        rows = [row for row in map(lambda p: front_upload(p, min_side), table.to_pylist()) if row]
        return group, rows, handle.fetched

    candidates: list[dict] = []
    seen: set[str] = set()
    used_groups = 0
    batch = workers * 8
    in_bucket = 0.9  # share of listed uploads the bucket holds; a first guess, measured below
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(candidates) < wanted and used_groups < len(order):
            picked: list[dict] = []
            # Read row groups until enough products are picked, then check them
            # against the bucket's file list in one pass (the list has 30 million lines).
            while len(candidates) + len(picked) * in_bucket < wanted and used_groups < len(order):
                groups = order[used_groups : used_groups + batch]
                used_groups += len(groups)
                # Results are taken in the sampled order, so the list does not depend on timing.
                for group, rows, size in pool.map(read_group, groups):
                    fetched += size
                    rows = [row for row in rows if row["Barcode"] not in seen]
                    random.Random(seed * 1_000_003 + group).shuffle(rows)
                    rows = rows[:PER_GROUP]
                    seen.update(row["Barcode"] for row in rows)
                    picked.extend(rows)
                print(f"  {used_groups} row groups read, {len(picked)} products picked", flush=True)
            print("  checking them against the bucket's file list", flush=True)
            present = keys_in_bucket(out, {row["Key"] for row in picked})
            kept = [row for row in picked if row["Key"] in present]
            candidates.extend(kept)
            in_bucket = max(0.05, len(kept) / max(1, len(picked))) * 0.98
            print(f"  {len(kept)} of {len(picked)} are in the bucket; {len(candidates)} candidates", flush=True)
    rng.shuffle(candidates)

    with open(out / "candidates.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)
    (out / "source.json").write_text(
        json.dumps(
            {
                "product_list": PARQUET_URL,
                "product_list_last_modified": remote.last_modified,
                "product_list_etag": remote.etag,
                "products_in_export": metadata.num_rows,
                "row_groups_read": used_groups,
                "bytes_read_from_export": fetched,
                "bucket": BUCKET_URL,
                "seed": seed,
                "min_side": min_side,
                "per_row_group": PER_GROUP,
                "sampled_on": datetime.date.today().isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {len(candidates)} candidates; {fetched / 1e6:.0f} MB read from the export")
    return candidates


def checked_size(path: Path, row: dict, min_side: int) -> None:
    """Raise Unusable unless the file is the image the list describes."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            size = image.size
            image.draft("RGB", (64, 64))  # decode small: this is a check that the data is whole
            image.load()
    except Exception as error:  # Pillow raises many types for a broken file
        raise Unusable(f"not a readable image: {type(error).__name__}") from error
    listed = (int(row["Width"]), int(row["Height"]))
    if size != listed and size != listed[::-1]:
        raise Unusable(f"is {size[0]}x{size[1]}, listed as {listed[0]}x{listed[1]}")
    if min(size) < min_side:
        raise Unusable(f"shorter side {min(size)} px")


def get_image(row: dict, images: Path, min_side: int) -> int:
    target = images / row["File"]
    written = fetch(BUCKET_URL + row["Key"], target)
    try:
        checked_size(target, row, min_side)
    except Unusable:
        target.unlink(missing_ok=True)
        raise
    return written


def is_complete(out: Path, count: int) -> bool:
    marker = out / "COMPLETE"
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == str(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="dataset folder; images go to <out>/images")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"images to fetch (default {DEFAULT_COUNT})")
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE, help="shorter side in pixels, at least (default 600)")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads from the bucket (default 16)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the product sample (default 0)")
    parser.add_argument("--list-only", action="store_true", help="sample the product list and stop before the images")
    parser.add_argument("--check", action="store_true", help="only report whether the dataset is already complete")
    args = parser.parse_args()

    out = Path(args.out)
    images = out / "images"
    if is_complete(out, args.count):
        print(f"dataset already complete: {args.count} images in {images}")
        return 0
    if args.check:
        print(f"dataset not complete in {out}")
        return 1

    print(f"dataset:  {NAME}; {args.count} products, one photo each, shorter side at least {args.min_side} px")
    print(f"licence:  CC BY-SA 3.0 ({LICENCE}); see THIRD_PARTY.md")
    started = time.time()
    images.mkdir(parents=True, exist_ok=True)

    candidates_path = out / "candidates.csv"
    if candidates_path.is_file():
        with open(candidates_path, "r", encoding="utf-8", newline="") as f:
            candidates = list(csv.DictReader(f))
        print(f"product list: {len(candidates)} candidates already sampled in {candidates_path}")
    else:
        candidates = sample_candidates(out, int(args.count * OVERSAMPLE) + 1, args.min_side, args.seed)
    if len(candidates) < args.count:
        print(f"only {len(candidates)} candidates for {args.count} images; delete candidates.csv to sample again for this count")
        return 1
    mean_bytes = sum(int(r["Width"]) * int(r["Height"]) for r in candidates) / len(candidates) * 0.25
    print(f"size:     roughly {args.count * mean_bytes / 1e9:.1f} GB (a guess from the pixel counts; the real figure is printed at the end)")
    if args.list_only:
        return 0

    skipped_path = out / "skipped.csv"
    skipped: dict[str, str] = {}
    if skipped_path.is_file():
        with open(skipped_path, "r", encoding="utf-8", newline="") as f:
            skipped = {row["File"]: row["Reason"] for row in csv.DictReader(f)}

    failures: list[str] = []
    downloaded = fetched_files = 0
    last_print = time.time()
    while True:
        usable = [row for row in candidates if row["File"] not in skipped]
        chosen = usable[: args.count]
        if len(chosen) < args.count:
            print(f"ran out of candidates: {len(chosen)} usable of {args.count} wanted")
            return 1
        todo = [row for row in chosen if not (images / row["File"]).is_file()]
        if not todo:
            break
        print(f"{args.count - len(todo)} images present, {len(todo)} to download with {args.workers} workers", flush=True)
        newly_skipped = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(get_image, row, images, args.min_side): row for row in todo}
            for done, job in enumerate(as_completed(jobs), start=1):
                row = jobs[job]
                try:
                    downloaded += job.result()
                    fetched_files += 1
                except Unusable as reason:
                    skipped[row["File"]] = str(reason)
                    newly_skipped += 1
                except RuntimeError as error:
                    failures.append(str(error))
                now = time.time()
                if now - last_print > 10 or done == len(todo):
                    last_print = now
                    rate = downloaded / max(1e-9, now - started)
                    print(f"  {done}/{len(todo)}  {downloaded / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s", flush=True)
        with open(skipped_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Reason"])
            writer.writerows(sorted(skipped.items()))
        if failures:
            print(f"\n{len(failures)} downloads failed, for example:\n  " + "\n  ".join(failures[:5]))
            print("Run the same command again to retry only the missing images.")
            return 1
        if newly_skipped:
            print(f"  {newly_skipped} candidates unusable (see {skipped_path}); taking the next ones on the list")

    total = 0
    with open(out / "ATTRIBUTION.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTRIBUTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in chosen:
            size = (images / row["File"]).stat().st_size
            total += size
            who = f"uploader {row['Uploader']}" if row["Uploader"] else "uploader not recorded"
            writer.writerow(
                {
                    **row,
                    "ProductPage": PRODUCT_PAGE.format(code=row["Barcode"]),
                    "ImageURL": BUCKET_URL + row["Key"],
                    "License": LICENCE,
                    "Credit": f"Open Food Facts contributors ({who}), https://openfoodfacts.org",
                    "Bytes": size,
                }
            )
    extra = sorted(p.name for p in images.iterdir() if p.name not in {row["File"] for row in chosen})
    if extra:
        print(f"note: {len(extra)} files in {images} are not on the list, for example {extra[:3]}")
    (out / "COMPLETE").write_text(f"{args.count}\n", encoding="utf-8")
    minutes = (time.time() - started) / 60
    print(f"done: {args.count} images, {total / 1e9:.2f} GB in {images}")
    print(f"this run: {fetched_files} images, {downloaded / 1e9:.2f} GB downloaded in {minutes:.1f} min")
    print(f"attribution for every image: {out / 'ATTRIBUTION.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
