#!/usr/bin/env python3
"""Experiment 06 - Data Preparation: RGB Full-Level
Step 1: Render level JSON → RGB image (uint8)
Step 2: Resize to fixed 64×512×3 using nearest-neighbor
"""
import json, os, numpy as np
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
    "glow": "skip", "fading": "skip", "particles": "skip",
    "clouds": "skip", "arrows": "skip", "text": "skip",
    "pixels": "skip", "hands": "skip", "monsters": "skip",
    "pickups": "skip", "collisions": "skip",
    "unknown": "unknown",
}

GROUP_COLORS = {
    "unknown":  (80, 80, 80),
    "blocks":   (160, 160, 160),
    "hazards":  (230, 50, 50),
    "triggers": (230, 210, 50),
    "portals":  (50, 220, 220),
    "orbs":     (50, 200, 80),
}
DRAW_ORDER = ["unknown", "blocks", "triggers", "orbs", "hazards", "portals"]
UNIT = 30
TH, TW = 64, 512

def resize_nearest(arr, th, tw):
    h, w, c = arr.shape
    ri = (np.arange(th) * h / th).astype(int).clip(0, h - 1)
    ci = (np.arange(tw) * w / tw).astype(int).clip(0, w - 1)
    return arr[np.ix_(ri, ci)]

def render_rgb(level_data, id_to_category, scale=1):
    objs = level_data.get("data", [])
    if not objs:
        return None
    points_by_group = {g: [] for g in DRAW_ORDER}
    for obj in objs:
        x, y, oid = obj.get("x"), obj.get("y"), obj.get("id")
        if x is None or y is None:
            continue
        cat = id_to_category.get(str(oid), "unknown") if oid else "unknown"
        group = CATEGORY_TO_GROUP.get(cat, "unknown")
        if group != "skip":
            points_by_group[group].append((x, y))
    all_x = [p[0] for v in points_by_group.values() for p in v]
    if not all_x:
        return None
    all_y = [p[1] for v in points_by_group.values() for p in v]
    x_min, x_max, y_min, y_max = min(all_x), max(all_x), min(all_y), max(all_y)
    w = int((x_max - x_min) / UNIT) + 1
    h = int((y_max - y_min) / UNIT) + 1
    img_w, img_h = w * scale, h * scale
    if img_w * img_h > 2e8:
        return None
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for group in DRAW_ORDER:
        color = GROUP_COLORS[group]
        for px, py in points_by_group[group]:
            ix = int((px - x_min) / UNIT)
            iy = int((y_max - py) / UNIT)
            x0, y0 = ix * scale, iy * scale
            img[y0:min(y0 + scale, img_h), x0:min(x0 + scale, img_w)] = color
    return img

def main():
    p = ArgumentParser()
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--stars", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    with open(os.path.join(PROJECT_ROOT, "src", "data", "id_to_category.json")) as f:
        idc = json.load(f)
    sr = [a.stars] if a.stars else range(1, 11)
    total = 0
    for ns in sr:
        dd = os.path.join(PROJECT_ROOT, "levels", f"{ns}stars")
        od = os.path.join(PROJECT_ROOT, "data", "levels_resized_rgb", f"{ns}stars")
        if not os.path.isdir(dd):
            continue
        os.makedirs(od, exist_ok=True)
        files = sorted(f for f in os.listdir(dd) if f.endswith(".json"))
        if a.limit:
            files = files[:a.limit]
        for fn in tqdm(files, desc=f"{ns}stars"):
            lid = fn.split(".")[0]
            op = os.path.join(od, f"{lid}.npy")
            if os.path.exists(op):
                total += 1
                continue
            with open(os.path.join(dd, fn)) as f:
                ld = json.load(f)
            img = render_rgb(ld, idc, scale=a.scale)
            if img is not None:
                np.save(op, resize_nearest(img, TH, TW))
                total += 1
    print(f"Done. {total} levels rendered + resized to {TH}x{TW}x3")

if __name__ == "__main__":
    main()
