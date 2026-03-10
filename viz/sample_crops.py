# Sample fixed-size crops from rendered level images

import os
import csv
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

# Resolve project root (one level up from viz/)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sample_crops_from_image(img_array, n, crop_w, crop_h, k, max_retries):
    """Sample n crops of crop_w x crop_h from img_array.
    Each crop must have at least k non-black pixels.
    Returns list of (crop_array, x, y, nonblack_count)."""
    h, w = img_array.shape[:2]

    # Pad if image is smaller than crop size
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w = img_array.shape[:2]

    max_x = w - crop_w
    max_y = h - crop_h

    crops = []
    for _ in range(n):
        for _ in range(max_retries):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            crop = img_array[y:y + crop_h, x:x + crop_w]
            nonblack = int(np.sum(crop.sum(axis=2) > 0))
            if nonblack >= k:
                crops.append((crop, x, y, nonblack))
                break
    return crops


def main():
    parser = ArgumentParser()
    parser.add_argument("--crop-w", type=int, default=128, help="Crop width (default: 128)")
    parser.add_argument("--crop-h", type=int, default=64, help="Crop height (default: 64)")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per level (default: 5)")
    parser.add_argument("--k", type=int, default=20, help="Min non-black pixels per crop (default: 20)")
    parser.add_argument("--max-retries", type=int, default=50, help="Max retries per sample (default: 50)")
    parser.add_argument("--stars", type=int, default=None, help="Only process this star rating")
    parser.add_argument("--limit", type=int, default=None, help="Max levels per star rating")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    star_range = [args.stars] if args.stars else range(1, 11)
    manifest = []

    for n_stars in star_range:
        in_dir = os.path.join(plots_dir, f"{n_stars}stars")
        out_dir = os.path.join(crops_dir, f"{n_stars}stars")
        if not os.path.isdir(in_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(in_dir) if f.endswith(".png")])
        if args.limit:
            files = files[:args.limit]

        print(f"Sampling {n_stars}stars: {len(files)} levels, {args.n} crops each (w={args.crop_w}, h={args.crop_h})...")
        skipped = 0
        for fname in tqdm(files):
            level_id = fname.split(".")[0]
            img = np.array(Image.open(os.path.join(in_dir, fname)))

            crops = sample_crops_from_image(
                img, args.n, args.crop_w, args.crop_h, args.k, args.max_retries
            )

            if len(crops) < args.n:
                skipped += 1

            for i, (crop, x, y, nonblack) in enumerate(crops):
                out_name = f"{level_id}_{i}.png"
                Image.fromarray(crop).save(os.path.join(out_dir, out_name))
                manifest.append({
                    "filename": f"{n_stars}stars/{out_name}",
                    "stars": n_stars,
                    "level_id": level_id,
                    "sample_idx": i,
                    "x": x,
                    "y": y,
                    "nonblack_pixels": nonblack,
                })

        if skipped:
            print(f"  {skipped} levels had fewer than {args.n} valid crops")

    # Write manifest
    os.makedirs(crops_dir, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "stars", "level_id", "sample_idx", "x", "y", "nonblack_pixels"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDone. Total crops: {len(manifest)}, saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
