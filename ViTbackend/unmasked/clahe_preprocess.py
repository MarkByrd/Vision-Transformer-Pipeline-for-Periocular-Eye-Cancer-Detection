"""
clahe_preprocess.py
-------------------
Applies CLAHE (Contrast Limited Adaptive Histogram Equalisation)
to the eyelid cancer dataset.

- Converts each image to LAB colour space
- Applies CLAHE to the L channel only (clip limit 2.0, tile grid 8×8)
- Preserves lesion colour information (A and B channels untouched)
- Saves to outputs/clahe_cancer/ preserving filenames
- Skips corrupt images (flagged in quality_report.csv if available)

Usage:
    python clahe_preprocess.py
    python clahe_preprocess.py --dataset "Eyelid Cancer Dataset" --clip 2.0
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULTS = dict(
    dataset=r"..\..\Eyelid_Cancer_Dataset",
    output=r"outputs\clahe_cancer",
    quality_report=r"outputs\quality_report.csv",
    clip=2.0,
    tile=8,
)


def load_corrupt_set(report_path: Path) -> set:
    """Load set of corrupt filenames from quality_report.csv (if exists)."""
    corrupt = set()
    if not report_path.exists():
        return corrupt
    with open(report_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("corrupt", "False").strip().lower() == "true":
                corrupt.add(row["filename"])
    return corrupt


def apply_clahe(bgr: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply CLAHE to L channel of LAB image. Returns BGR."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid)
    )
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return bgr_eq


def process_image(src: Path, dst: Path, clip: float, tile: int) -> str:
    """Process a single image. Returns 'ok', 'skip', or 'error:<msg>'."""
    bgr = cv2.imread(str(src))
    if bgr is None:
        return "error:cv2 read failed"

    try:
        bgr_eq = apply_clahe(bgr, clip, tile)
        # Always save as JPEG (normalise extension to .jpg)
        out_name = dst.stem + ".jpg"
        out_path = dst.parent / out_name
        success = cv2.imwrite(str(out_path), bgr_eq, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return "error:cv2 write failed"
        return "ok"
    except Exception as e:
        return f"error:{e}"


def main():
    parser = argparse.ArgumentParser(
        description="Apply CLAHE to the eyelid cancer dataset."
    )
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--quality_report", default=DEFAULTS["quality_report"])
    parser.add_argument("--clip", type=float, default=DEFAULTS["clip"],
                        help="CLAHE clip limit (default 2.0)")
    parser.add_argument("--tile", type=int, default=DEFAULTS["tile"],
                        help="CLAHE tile grid size (default 8)")
    args = parser.parse_args()

    src_root = Path(args.dataset)
    dst_root = Path(args.output)
    report_path = Path(args.quality_report)

    if not src_root.exists():
        print(f"ERROR: Dataset path not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)

    # Load corrupt list
    corrupt_set = load_corrupt_set(report_path)
    if corrupt_set:
        print(f"[INFO] Skipping {len(corrupt_set)} corrupt file(s) from quality_report.")

    images = sorted([
        f for f in src_root.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        print(f"No images found in {src_root}")
        sys.exit(0)

    print(f"\nCLAHE Preprocessing")
    print(f"  Source      : {src_root}")
    print(f"  Destination : {dst_root}")
    print(f"  Clip limit  : {args.clip}")
    print(f"  Tile grid   : {args.tile}x{args.tile}")
    print(f"  Images      : {len(images)}\n")

    n_ok = 0
    n_skip = 0
    n_err = 0

    for i, img_path in enumerate(images, 1):
        if img_path.name in corrupt_set:
            print(f"  [{i:>3}/{len(images)}] SKIP (corrupt)  {img_path.name}")
            n_skip += 1
            continue

        dst_path = dst_root / img_path.name
        status = process_image(img_path, dst_path, args.clip, args.tile)

        if status == "ok":
            n_ok += 1
            print(f"  [{i:>3}/{len(images)}] OK             {img_path.name}")
        elif status == "skip":
            n_skip += 1
            print(f"  [{i:>3}/{len(images)}] SKIP           {img_path.name}")
        else:
            n_err += 1
            print(f"  [{i:>3}/{len(images)}] ERROR: {status}   {img_path.name}")

    print(f"\n{'='*55}")
    print(f"  CLAHE COMPLETE")
    print(f"  Processed : {n_ok}")
    print(f"  Skipped   : {n_skip}")
    print(f"  Errors    : {n_err}")
    print(f"  Output    : {dst_root}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
