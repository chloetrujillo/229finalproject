# Render GD levels as 6-channel images with per-type scaling
# Each object type gets its own binary channel + independent scale multiplier.
# Sparse types (hazards, orbs, portals) are rendered with larger stamps
# so they survive CNN spatial downsampling.
#
# Output: .npy files with shape (H, W, 6), float32, values 0.0 or 1.0
# Channel order: hazards, blocks, triggers, portals, orbs, unknown

import json
import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Category -> group (same as render_levels.py)
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
    "glow": "skip", "fading": "skip", "particles": "skip",
    "clouds": "skip", "arrows": "skip", "text": "skip",
    "pixels": "skip", "hands": "skip", "monsters": "skip",
    "pickups": "skip", "collisions": "skip",
    "unknown": "unknown",
}

# Channel order
CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# Per-type scale multiplier (relative to base scale)
# Sparse types get larger stamps so they survive 8x downsampling
TYPE_WEIGHT = {
    "hazards":  3,   # 3x → with base scale=2: 6×6 stamp
    "blocks":   1,   # 1x → 2×2 stamp
    "triggers": 1,   # 1x → 2×2 stamp (already ~0.8%, enough)
    "portals":  3,   # 3x → 6×6 stamp
    "orbs":     3,   # 3x → 6×6 stamp
    "unknown":  1,   # 1x → 2×2 stamp
}

UNIT = 30  # GD units per pixel


def render_level_weighted(level_data, id_to_category, scale=2):
    """Render a level as a 6-channel numpy array with per-type scaling.

    Returns: (H, W, 6) float32 array, or None if empty.
    """
    objs = level_data.get("data", [])
    if not objs:
        return None

    # Collect points by group
    points_by_group = {g: [] for g in CHANNEL_NAMES}
    for obj in objs:
        x = obj.get("x")
        y = obj.get("y")
        obj_id = obj.get("id")
        if x is None or y is None:
            continue
        cat = id_to_category.get(str(obj_id), "unknown") if obj_id is not None else "unknown"
        group = CATEGORY_TO_GROUP.get(cat, "unknown")
        if group == "skip":
            continue
        points_by_group[group].append((x, y))

    # Compute bounds from all points
    all_x, all_y = [], []
    for pts in points_by_group.values():
        for px, py in pts:
            all_x.append(px)
            all_y.append(py)

    if not all_x:
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    w = int((x_max - x_min) / UNIT) + 1
    h = int((y_max - y_min) / UNIT) + 1

    # Compute max stamp size to determine padding needed
    max_weight = max(TYPE_WEIGHT.values())
    max_stamp = scale * max_weight
    pad = max_stamp // 2  # extra padding for large stamps

    img_w = w * scale + 2 * pad
    img_h = h * scale + 2 * pad

    if img_w * img_h > 200_000_000:
        return None

    img = np.zeros((img_h, img_w, 6), dtype=np.float32)

    # Render each group to its own channel
    for group in CHANNEL_NAMES:
        ch_idx = CHANNEL_INDEX[group]
        weight = TYPE_WEIGHT[group]
        stamp_size = scale * weight

        for px, py in points_by_group[group]:
            ix = int((px - x_min) / UNIT)
            iy = int((y_max - py) / UNIT)  # flip y

            # Center the stamp on the base-scale position
            base_cx = pad + ix * scale + scale // 2
            base_cy = pad + iy * scale + scale // 2

            x0 = max(0, base_cx - stamp_size // 2)
            y0 = max(0, base_cy - stamp_size // 2)
            x1 = min(img_w, x0 + stamp_size)
            y1 = min(img_h, y0 + stamp_size)

            img[y0:y1, x0:x1, ch_idx] = 1.0

    # Trim padding (remove rows/cols that are all zero)
    any_content = img.sum(axis=2) > 0
    rows = np.any(any_content, axis=1)
    cols = np.any(any_content, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        img = img[r0:r1+1, c0:c1+1]

    return img


def main():
    parser = ArgumentParser()
    parser.add_argument("--scale", type=int, default=2, help="Base pixel size per object (default: 2)")
    parser.add_argument("--stars", type=int, default=None, help="Only render this star rating")
    parser.add_argument("--limit", type=int, default=None, help="Max levels per star rating")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, "src", "data", "id_to_category.json"), "r") as f:
        id_to_category = json.load(f)

    star_range = [args.stars] if args.stars else range(1, 11)

    print(f"Rendering with base scale={args.scale}")
    print(f"Type weights: {TYPE_WEIGHT}")
    print(f"Stamp sizes: { {k: args.scale * v for k, v in TYPE_WEIGHT.items()} }")

    for n in star_range:
        data_dir = os.path.join(PROJECT_ROOT, "levels", f"{n}stars")
        out_dir = os.path.join(PROJECT_ROOT, "plots_weighted", f"{n}stars")
        if not os.path.isdir(data_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        if args.limit:
            files = files[:args.limit]

        print(f"Rendering {len(files)} levels for {n} stars...")
        for fname in tqdm(files):
            level_id = fname.split(".")[0]
            out_path = os.path.join(out_dir, f"{level_id}.npy")
            if os.path.exists(out_path):
                continue

            with open(os.path.join(data_dir, fname), "r") as f:
                level_data = json.load(f)

            img = render_level_weighted(level_data, id_to_category, scale=args.scale)
            if img is not None:
                np.save(out_path, img)

    print("Done.")


if __name__ == "__main__":
    main()
