# Render GD levels as images with semantic color groups

import json
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

# Resolve project root (one level up from viz/)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Category -> color group
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

# Group -> RGB color
GROUP_COLORS = {
    "unknown":    (80, 80, 80),       # dark gray
    "blocks":     (160, 160, 160),     # gray
    "hazards":    (230, 50, 50),       # red
    "triggers":   (230, 210, 50),      # yellow
    "portals":    (50, 220, 220),      # cyan
    "orbs":       (50, 200, 80),       # green
}

# Draw order: background first, important elements last
DRAW_ORDER = ["unknown", "blocks", "triggers", "orbs", "hazards", "portals"]


def render_level(level_data, id_to_category, scale=1):
    """Render a level as a PIL Image."""
    objs = level_data.get("data", [])
    if not objs:
        return None

    # Collect (x, y, group) for each object
    points_by_group = {g: [] for g in DRAW_ORDER}

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

    # Compute bounds
    all_x = []
    all_y = []
    for pts in points_by_group.values():
        for px, py in pts:
            all_x.append(px)
            all_y.append(py)

    if not all_x:
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Normalize to pixel coordinates
    UNIT = 30  # GD units per pixel (compress the coordinate space)
    w = int((x_max - x_min) / UNIT) + 1
    h = int((y_max - y_min) / UNIT) + 1

    # Apply scale
    img_w = w * scale
    img_h = h * scale

    # Cap image size to avoid memory issues
    if img_w * img_h > 200_000_000:
        return None

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Draw in order
    for group in DRAW_ORDER:
        color = GROUP_COLORS[group]
        for px, py in points_by_group[group]:
            ix = int((px - x_min) / UNIT)
            # Flip y: GD y increases upward, image y increases downward
            iy = int((y_max - py) / UNIT)
            # Draw scale x scale block
            x0 = ix * scale
            y0 = iy * scale
            x1 = min(x0 + scale, img_w)
            y1 = min(y0 + scale, img_h)
            img[y0:y1, x0:x1] = color

    return Image.fromarray(img)


def main():
    parser = ArgumentParser()
    parser.add_argument("--scale", type=int, default=1, help="Pixel size per object (default: 1)")
    parser.add_argument("--stars", type=int, default=None, help="Only render levels of this star rating")
    parser.add_argument("--limit", type=int, default=None, help="Max levels to render per star rating")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, "src", "data", "id_to_category.json"), "r") as f:
        id_to_category = json.load(f)

    star_range = [args.stars] if args.stars else range(1, 11)

    for n in star_range:
        data_dir = os.path.join(PROJECT_ROOT, "levels", f"{n}stars")
        out_dir = os.path.join(PROJECT_ROOT, "plots", f"{n}stars")
        if not os.path.isdir(data_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        if args.limit:
            files = files[:args.limit]

        print(f"Rendering {len(files)} levels for {n} stars...")
        for fname in tqdm(files):
            level_id = fname.split(".")[0]
            out_path = os.path.join(out_dir, f"{level_id}.png")
            if os.path.exists(out_path):
                continue

            with open(os.path.join(data_dir, fname), "r") as f:
                level_data = json.load(f)

            img = render_level(level_data, id_to_category, scale=args.scale)
            if img is not None:
                img.save(out_path)


if __name__ == "__main__":
    main()
