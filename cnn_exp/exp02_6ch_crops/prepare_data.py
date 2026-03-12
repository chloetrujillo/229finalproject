#!/usr/bin/env python3
"""Experiment 02 - Data Preparation: 6-Channel Crops (same as exp01)
Renders RGB PNGs + samples crops. The 6-channel conversion happens at training time.
See exp01/prepare_data.py for full documentation.
"""
# This experiment uses the SAME data as exp01. The 6-channel conversion
# (RGB color → binary channel per object type) happens in train.py at load time.
# Run exp01/prepare_data.py to generate the data, or copy this identical pipeline.

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
    "unknown": (80,80,80), "blocks": (160,160,160), "hazards": (230,50,50),
    "triggers": (230,210,50), "portals": (50,220,220), "orbs": (50,200,80),
}
DRAW_ORDER = ["unknown", "blocks", "triggers", "orbs", "hazards", "portals"]
UNIT = 30

def render_level(level_data, id_to_category, scale=1):
    objs = level_data.get("data", [])
    if not objs: return None
    points = {g: [] for g in DRAW_ORDER}
    for obj in objs:
        x, y, oid = obj.get("x"), obj.get("y"), obj.get("id")
        if x is None or y is None: continue
        cat = id_to_category.get(str(oid), "unknown") if oid else "unknown"
        g = CATEGORY_TO_GROUP.get(cat, "unknown")
        if g != "skip": points[g].append((x, y))
    ax = [p[0] for pts in points.values() for p in pts]
    ay = [p[1] for pts in points.values() for p in pts]
    if not ax: return None
    xn, xx, yn, yx = min(ax), max(ax), min(ay), max(ay)
    w, h = int((xx-xn)/UNIT)+1, int((yx-yn)/UNIT)+1
    iw, ih = w*scale, h*scale
    if iw*ih > 200_000_000: return None
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    for g in DRAW_ORDER:
        c = GROUP_COLORS[g]
        for px, py in points[g]:
            ix, iy = int((px-xn)/UNIT), int((yx-py)/UNIT)
            x0, y0 = ix*scale, iy*scale
            img[y0:min(y0+scale,ih), x0:min(x0+scale,iw)] = c
    return Image.fromarray(img)

def sample_crops(img_arr, n, cw, ch, k, retries=50):
    h, w = img_arr.shape[:2]
    ph, pw = max(0, ch-h), max(0, cw-w)
    if ph or pw: img_arr = np.pad(img_arr, ((0,ph),(0,pw),(0,0)), mode='constant'); h,w = img_arr.shape[:2]
    crops = []
    for _ in range(n):
        for _ in range(retries):
            x, y = random.randint(0, w-cw), random.randint(0, h-ch)
            crop = img_arr[y:y+ch, x:x+cw]
            nb = int(np.sum(crop.sum(axis=2) > 0))
            if nb >= k: crops.append((crop, x, y, nb)); break
    return crops

def main():
    parser = ArgumentParser()
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--crop-w", type=int, default=128)
    parser.add_argument("--crop-h", type=int, default=64)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--stars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    with open(os.path.join(PROJECT_ROOT, "src", "data", "id_to_category.json")) as f:
        id_to_cat = json.load(f)

    star_range = [args.stars] if args.stars else range(1, 11)
    crops_dir = os.path.join(PROJECT_ROOT, "data", "crops")
    manifest = []

    for ns in star_range:
        dd = os.path.join(PROJECT_ROOT, "levels", f"{ns}stars")
        pd = os.path.join(PROJECT_ROOT, "data", "plots", f"{ns}stars")
        cd = os.path.join(crops_dir, f"{ns}stars")
        if not os.path.isdir(dd): continue
        os.makedirs(pd, exist_ok=True); os.makedirs(cd, exist_ok=True)
        files = sorted(f for f in os.listdir(dd) if f.endswith(".json"))
        if args.limit: files = files[:args.limit]
        for fn in tqdm(files, desc=f"{ns}stars"):
            lid = fn.split(".")[0]
            pp = os.path.join(pd, f"{lid}.png")
            if not os.path.exists(pp):
                with open(os.path.join(dd, fn)) as f: ld = json.load(f)
                img = render_level(ld, id_to_cat, scale=args.scale)
                if img is None: continue
                img.save(pp)
            else: img = Image.open(pp)
            for i, (crop, x, y, nb) in enumerate(sample_crops(np.array(img), args.n, args.crop_w, args.crop_h, args.k)):
                on = f"{lid}_{i}.png"
                Image.fromarray(crop).save(os.path.join(cd, on))
                manifest.append({"filename": f"{ns}stars/{on}", "stars": ns, "level_id": lid,
                                "sample_idx": i, "x": x, "y": y, "nonblack_pixels": nb})

    mp = os.path.join(crops_dir, "manifest.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename","stars","level_id","sample_idx","x","y","nonblack_pixels"])
        w.writeheader(); w.writerows(manifest)
    print(f"Done. {len(manifest)} crops")

if __name__ == "__main__":
    main()
