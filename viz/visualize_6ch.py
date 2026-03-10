# Visualize 6-channel decomposition of merged crops
# Shows original RGB + each object-type channel side by side

import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CROP_W = 128
CROP_H = 64
WINDOW = 3
PAD = 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)

# From render_levels.py
GROUP_COLORS_U8 = {
    "hazards":  np.array([230, 50, 50], dtype=np.uint8),
    "blocks":   np.array([160, 160, 160], dtype=np.uint8),
    "triggers": np.array([230, 210, 50], dtype=np.uint8),
    "portals":  np.array([50, 220, 220], dtype=np.uint8),
    "orbs":     np.array([50, 200, 80], dtype=np.uint8),
    "unknown":  np.array([80, 80, 80], dtype=np.uint8),
}
CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
CHANNEL_CMAPS = ["Reds", "Greys", "YlOrBr", "cool", "Greens", "Purples"]


def rgb_to_6ch(img_u8):
    h, w = img_u8.shape[:2]
    out = np.zeros((h, w, len(CHANNEL_NAMES)), dtype=np.float32)
    for i, name in enumerate(CHANNEL_NAMES):
        color = GROUP_COLORS_U8[name]
        match = np.all(img_u8 == color[np.newaxis, np.newaxis, :], axis=2)
        out[:, :, i] = match.astype(np.float32)
    return out


def merge_crops(imgs, start_idx=0):
    parts = []
    n_ch = imgs[0].shape[2]
    for j in range(WINDOW):
        if j > 0:
            parts.append(np.zeros((CROP_H, PAD, n_ch), dtype=imgs[0].dtype))
        idx = start_idx + j
        if idx < len(imgs):
            parts.append(imgs[idx])
        else:
            parts.append(np.zeros((CROP_H, CROP_W, n_ch), dtype=imgs[0].dtype))
    return np.concatenate(parts, axis=1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3, help="Number of levels")
    parser.add_argument("--merge-idx", type=int, default=0)
    args = parser.parse_args()

    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))

    level_map = defaultdict(list)
    level_stars = {}
    for r in all_rows:
        if r["stars"] == str(args.stars):
            level_map[r["level_id"]].append(r)
            level_stars[r["level_id"]] = r["stars"]

    level_ids = sorted(level_map.keys())[:args.limit]
    out_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)

    for lid in level_ids:
        rows = sorted(level_map[lid], key=lambda r: int(r["x"]))

        # Load RGB crops
        rgb_imgs = []
        for r in rows:
            img = np.array(Image.open(os.path.join(crops_dir, r["filename"])))
            rgb_imgs.append(img)

        # Convert to 6ch
        ch6_imgs = [rgb_to_6ch(img) for img in rgb_imgs]

        # Merge
        merge_start = min(args.merge_idx, max(0, len(rgb_imgs) - WINDOW))
        merged_rgb = merge_crops(rgb_imgs, merge_start)
        merged_6ch = merge_crops(ch6_imgs, merge_start)

        # Plot: 1 row for RGB + 6 rows for channels = 7 rows
        fig, axes = plt.subplots(7, 1, figsize=(14, 10))

        # Row 0: original RGB
        axes[0].imshow(merged_rgb)
        axes[0].set_title("Original RGB", fontsize=10)
        axes[0].set_ylabel("RGB", fontsize=9)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # Rows 1-6: each channel
        for i, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
            ch = merged_6ch[:, :, i]
            px_count = int(ch.sum())
            pct = px_count / (MERGED_W * CROP_H) * 100

            axes[i + 1].imshow(ch, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            axes[i + 1].set_title(f"ch{i}: {name}  ({px_count} px, {pct:.1f}%)", fontsize=10)
            axes[i + 1].set_ylabel(name, fontsize=8)
            axes[i + 1].set_xticks([]); axes[i + 1].set_yticks([])

        crop_ids = [rows[merge_start + k]["sample_idx"] if merge_start + k < len(rows) else "pad"
                    for k in range(WINDOW)]
        fig.suptitle(f"6-Channel Decomposition: {lid} ({level_stars[lid]}★) "
                     f"crops [{','.join(crop_ids)}]", fontsize=12, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"6ch_vis_{lid}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
