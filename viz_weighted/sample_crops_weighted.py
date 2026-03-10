# Sample fixed-size crops from 6-channel weighted level images (.npy)
# Output: .npy crop files (H, W, 6) + manifest.csv

import os
import csv
import random
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sample_crops_from_6ch(img_6ch, n, crop_w, crop_h, k, max_retries):
    """Sample n crops of crop_w x crop_h from a 6-channel image.
    Each crop must have at least k non-zero pixels (any channel).
    Returns list of (crop_array, x, y, nonzero_count)."""
    h, w = img_6ch.shape[:2]

    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h > 0 or pad_w > 0:
        img_6ch = np.pad(img_6ch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w = img_6ch.shape[:2]

    max_x = w - crop_w
    max_y = h - crop_h

    crops = []
    for _ in range(n):
        for _ in range(max_retries):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            crop = img_6ch[y:y + crop_h, x:x + crop_w]
            nonzero = int((crop.sum(axis=2) > 0).sum())
            if nonzero >= k:
                crops.append((crop, x, y, nonzero))
                break
    return crops


def main():
    parser = ArgumentParser()
    parser.add_argument("--crop-w", type=int, default=128)
    parser.add_argument("--crop-h", type=int, default=64)
    parser.add_argument("--n", type=int, default=10, help="Samples per level")
    parser.add_argument("--k", type=int, default=40, help="Min non-zero pixels per crop")
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument("--stars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    plots_dir = os.path.join(PROJECT_ROOT, "plots_weighted")
    crops_dir = os.path.join(PROJECT_ROOT, "crops_weighted")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    star_range = [args.stars] if args.stars else range(1, 11)
    manifest = []

    for n_stars in star_range:
        in_dir = os.path.join(plots_dir, f"{n_stars}stars")
        out_dir = os.path.join(crops_dir, f"{n_stars}stars")
        if not os.path.isdir(in_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npy")])
        if args.limit:
            files = files[:args.limit]

        print(f"Sampling {n_stars}stars: {len(files)} levels, {args.n} crops each "
              f"(w={args.crop_w}, h={args.crop_h})...")
        skipped = 0
        for fname in tqdm(files):
            level_id = fname.split(".")[0]
            img_6ch = np.load(os.path.join(in_dir, fname))

            crops = sample_crops_from_6ch(
                img_6ch, args.n, args.crop_w, args.crop_h, args.k, args.max_retries
            )

            if len(crops) < args.n:
                skipped += 1

            for i, (crop, x, y, nonzero) in enumerate(crops):
                out_name = f"{level_id}_{i}.npy"
                np.save(os.path.join(out_dir, out_name), crop)
                manifest.append({
                    "filename": f"{n_stars}stars/{out_name}",
                    "stars": n_stars,
                    "level_id": level_id,
                    "sample_idx": i,
                    "x": x,
                    "y": y,
                    "nonzero_pixels": nonzero,
                })

        if skipped:
            print(f"  {skipped} levels had fewer than {args.n} valid crops")

    os.makedirs(crops_dir, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "stars", "level_id", "sample_idx", "x", "y", "nonzero_pixels"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDone. Total crops: {len(manifest)}, saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
