#!/usr/bin/env python3
"""Experiment 01 - Data Preparation: RGB Crops
Renders level JSONs as RGB images, then samples fixed-size crops.
"""

import json, os, csv, random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CATEGORY_TO_GROUP = {
    "block": "blocks",
    "spikes": "hazards", "saws": "hazards", "fire": "hazards", "breakable": "hazards",
    "trigger_color": "triggers", "trigger_move": "triggers", "trigger_pulse": "triggers",
    "trigger_alpha": "triggers", "trigger_spawn": "triggers", "trigger_rotate": "triggers",
    "trigger_follow": "triggers", "trigger_shake": "triggers", "trigger_toggle": "triggers",
    "trigger_transition": "triggers", "trigger_startpos": "triggers", "trigger_other": "triggers",
    "portal_gamemode": "portals", "portal_speed": "portals", "portal_size": "portals",
    "portal_mirror": "portals", "portal_dual": "portals",
    "orb": "orbs", "pulsing": "orbs", "coin": "orbs",
    "glow": "skip", "fading": "skip", "particles": "skip", "clouds": "skip",
    "arrows": "skip", "text": "skip", "pixels": "skip", "hands": "skip",
    "monsters": "skip", "pickups": "skip", "collisions": "skip",
    "unknown": "unknown",
}

GROUP_COLORS = {
    "unknown": (80, 80, 80), "blocks": (160, 160, 160), "hazards": (230, 50, 50),
    "triggers": (230, 210, 50), "portals": (50, 220, 220), "orbs": (50, 200, 80),
}
DRAW_ORDER = ["unknown", "blocks", "triggers", "orbs", "hazards", "portals"]
UNIT = 30


def render_level(level_data, id_to_category, scale=1):
    objs = level_data.get("data", [])
    if not objs:
        return None
    points_by_group = {g: [] for g in DRAW_ORDER}
    for obj in objs:
        x, y, obj_id = obj.get("x"), obj.get("y"), obj.get("id")
        if x is None or y is None:
            continue
        cat = id_to_category.get(str(obj_id), "unknown") if obj_id else "unknown"
        group = CATEGORY_TO_GROUP.get(cat, "unknown")
        if group != "skip":
            points_by_group[group].append((x, y))

    all_x = [p[0] for pts in points_by_group.values() for p in pts]
    all_y = [p[1] for pts in points_by_group.values() for p in pts]
    if not all_x:
        return None

    x_min, x_max, y_min, y_max = min(all_x), max(all_x), min(all_y), max(all_y)
    w = int((x_max - x_min) / UNIT) + 1
    h = int((y_max - y_min) / UNIT) + 1
    img_w, img_h = w * scale, h * scale
    if img_w * img_h > 200_000_000:
        return None

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for group in DRAW_ORDER:
        color = GROUP_COLORS[group]
        for px, py in points_by_group[group]:
            ix = int((px - x_min) / UNIT)
            iy = int((y_max - py) / UNIT)
            x0, y0 = ix * scale, iy * scale
            x1, y1 = min(x0 + scale, img_w), min(y0 + scale, img_h)
            img[y0:y1, x0:x1] = color
    return Image.fromarray(img)


def sample_crops(img_array, n, crop_w, crop_h, k, max_retries=50):
    h, w = img_array.shape[:2]
    pad_h, pad_w = max(0, crop_h - h), max(0, crop_w - w)
    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w = img_array.shape[:2]
    crops = []
    for _ in range(n):
        for _ in range(max_retries):
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            crop = img_array[y:y + crop_h, x:x + crop_w]
            nonblack = int(np.sum(crop.sum(axis=2) > 0))
            if nonblack >= k:
                crops.append((crop, x, y, nonblack))
                break
    return crops


def main():
    parser = ArgumentParser()
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Crops per level")
    parser.add_argument("--crop-w", type=int, default=128)
    parser.add_argument("--crop-h", type=int, default=64)
    parser.add_argument("--k", type=int, default=20, help="Min non-black pixels")
    parser.add_argument("--stars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(PROJECT_ROOT, "src", "data", "id_to_category.json")) as f:
        id_to_category = json.load(f)

    star_range = [args.stars] if args.stars else range(1, 11)
    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    crops_dir = os.path.join(PROJECT_ROOT, "data", "crops")
    manifest = []

    for n_stars in star_range:
        data_dir = os.path.join(PROJECT_ROOT, "levels", f"{n_stars}stars")
        plot_dir = os.path.join(plots_dir, f"{n_stars}stars")
        crop_dir = os.path.join(crops_dir, f"{n_stars}stars")
        if not os.path.isdir(data_dir):
            continue
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(crop_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
        if args.limit:
            files = files[:args.limit]
        print(f"{n_stars}stars: {len(files)} levels")

        for fname in tqdm(files, desc=f"{n_stars}stars"):
            level_id = fname.split(".")[0]
            plot_path = os.path.join(plot_dir, f"{level_id}.png")

            # Step 1: Render
            if not os.path.exists(plot_path):
                with open(os.path.join(data_dir, fname)) as f:
                    level_data = json.load(f)
                img = render_level(level_data, id_to_category, scale=args.scale)
                if img is None:
                    continue
                img.save(plot_path)
            else:
                img = Image.open(plot_path)

            # Step 2: Crop
            img_arr = np.array(img)
            crops = sample_crops(img_arr, args.n, args.crop_w, args.crop_h, args.k)
            for i, (crop, x, y, nb) in enumerate(crops):
                out_name = f"{level_id}_{i}.png"
                Image.fromarray(crop).save(os.path.join(crop_dir, out_name))
                manifest.append({"filename": f"{n_stars}stars/{out_name}", "stars": n_stars,
                                "level_id": level_id, "sample_idx": i, "x": x, "y": y,
                                "nonblack_pixels": nb})

    manifest_path = os.path.join(crops_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename","stars","level_id","sample_idx","x","y","nonblack_pixels"])
        w.writeheader()
        w.writerows(manifest)
    print(f"Done. {len(manifest)} crops, manifest: {manifest_path}")


if __name__ == "__main__":
    main()
