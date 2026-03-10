# Visualize weighted 6-channel crops vs original RGB crops side by side

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

CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
# Display colors for each channel (to reconstruct a colored overlay)
DISPLAY_COLORS = {
    "hazards":  [1.0, 0.2, 0.2],
    "blocks":   [0.63, 0.63, 0.63],
    "triggers": [0.9, 0.82, 0.2],
    "portals":  [0.2, 0.86, 0.86],
    "orbs":     [0.2, 0.78, 0.31],
    "unknown":  [0.31, 0.31, 0.31],
}
CHANNEL_CMAPS = ["Reds", "Greys", "YlOrBr", "cool", "Greens", "Purples"]


def merge_crops_npy(imgs, start_idx=0):
    n_ch = imgs[0].shape[2]
    parts = []
    for j in range(WINDOW):
        if j > 0:
            parts.append(np.zeros((CROP_H, PAD, n_ch), dtype=np.float32))
        idx = start_idx + j
        if idx < len(imgs):
            parts.append(imgs[idx])
        else:
            parts.append(np.zeros((CROP_H, CROP_W, n_ch), dtype=np.float32))
    return np.concatenate(parts, axis=1)


def merge_crops_rgb(imgs, start_idx=0):
    parts = []
    for j in range(WINDOW):
        if j > 0:
            parts.append(np.zeros((CROP_H, PAD, 3), dtype=np.uint8))
        idx = start_idx + j
        if idx < len(imgs):
            parts.append(imgs[idx])
        else:
            parts.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def channels_to_rgb(img_6ch):
    """Convert 6-channel binary image to colored RGB for display."""
    h, w = img_6ch.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Draw in order: unknown first, hazards last (on top)
    draw_order = [5, 1, 2, 4, 0, 3]  # unknown, blocks, triggers, orbs, hazards, portals
    for idx in draw_order:
        name = CHANNEL_NAMES[idx]
        color = np.array(DISPLAY_COLORS[name])
        mask = img_6ch[:, :, idx] > 0.5
        rgb[mask] = color
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--merge-idx", type=int, default=0)
    args = parser.parse_args()

    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    crops_w_dir = os.path.join(PROJECT_ROOT, "crops_weighted")
    manifest_rgb = os.path.join(crops_dir, "manifest.csv")
    manifest_w = os.path.join(crops_w_dir, "manifest.csv")

    # Check if weighted crops exist
    if not os.path.exists(manifest_w):
        print(f"No weighted crops found. Run render_weighted.py + sample_crops_weighted.py first.")
        print(f"Falling back: showing weighted full-level images instead.")
        # Show full-level .npy files
        plots_dir = os.path.join(PROJECT_ROOT, "plots_weighted", f"{args.stars}stars")
        if not os.path.isdir(plots_dir):
            print(f"No plots_weighted/{args.stars}stars/ found either. Run render_weighted.py first.")
            return
        files = sorted(os.listdir(plots_dir))[:args.limit]
        for fname in files:
            img_6ch = np.load(os.path.join(plots_dir, fname))
            lid = fname.split(".")[0]

            # Crop a 386×64 region for visualization
            h, w = img_6ch.shape[:2]
            cx = min(w - CROP_W * WINDOW, max(0, w // 4))
            cy = max(0, h // 2 - CROP_H // 2)
            region = img_6ch[cy:cy+CROP_H, cx:cx+CROP_W*WINDOW]

            fig, axes = plt.subplots(8, 1, figsize=(14, 11))
            # Row 0: colored overlay
            rgb_overlay = channels_to_rgb(region)
            axes[0].imshow(rgb_overlay)
            axes[0].set_title("Weighted 6ch → colored overlay", fontsize=10)
            axes[0].set_ylabel("overlay", fontsize=8)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            # Row 1: original RGB (from plots/)
            orig_path = os.path.join(PROJECT_ROOT, "plots", f"{args.stars}stars", f"{lid}.png")
            if os.path.exists(orig_path):
                orig = np.array(Image.open(orig_path))
                oh, ow = orig.shape[:2]
                # Scale crop position
                ox = min(ow - CROP_W * WINDOW, max(0, ow // 4))
                oy = max(0, oh // 2 - CROP_H // 2)
                orig_region = orig[oy:oy+CROP_H, ox:ox+CROP_W*WINDOW]
                axes[1].imshow(orig_region)
                axes[1].set_title("Original RGB (same region approx)", fontsize=10)
            else:
                axes[1].text(0.5, 0.5, "Original not found", transform=axes[1].transAxes, ha='center')
            axes[1].set_ylabel("original", fontsize=8)
            axes[1].set_xticks([]); axes[1].set_yticks([])

            # Rows 2-7: individual channels
            for i, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
                ch = region[:, :, i]
                px = int(ch.sum())
                axes[i+2].imshow(ch, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                axes[i+2].set_title(f"ch{i}: {name} ({px} px)", fontsize=9)
                axes[i+2].set_ylabel(name, fontsize=8)
                axes[i+2].set_xticks([]); axes[i+2].set_yticks([])

            fig.suptitle(f"Weighted 6ch: {lid} ({args.stars}★)", fontsize=12, y=1.01)
            plt.tight_layout()
            out_path = os.path.join(PROJECT_ROOT, "models", f"weighted_vis_{lid}.png")
            os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")
        return

    # If weighted crops exist, show merged crops comparison
    with open(manifest_rgb) as f:
        rgb_rows = list(csv.DictReader(f))
    with open(manifest_w) as f:
        w_rows = list(csv.DictReader(f))

    # Group by level
    rgb_map = defaultdict(list)
    w_map = defaultdict(list)
    for r in rgb_rows:
        if r["stars"] == str(args.stars):
            rgb_map[r["level_id"]].append(r)
    for r in w_rows:
        if r["stars"] == str(args.stars):
            w_map[r["level_id"]].append(r)

    common_ids = sorted(set(rgb_map.keys()) & set(w_map.keys()))[:args.limit]

    for lid in common_ids:
        rgb_sorted = sorted(rgb_map[lid], key=lambda r: int(r["x"]))
        w_sorted = sorted(w_map[lid], key=lambda r: int(r["x"]))

        # Load and merge RGB crops
        rgb_imgs = [np.array(Image.open(os.path.join(crops_dir, r["filename"])))
                    for r in rgb_sorted]
        # Load and merge weighted crops
        w_imgs = [np.load(os.path.join(crops_w_dir, r["filename"]))
                  for r in w_sorted]

        ms = min(args.merge_idx, max(0, len(rgb_imgs) - WINDOW), max(0, len(w_imgs) - WINDOW))
        merged_rgb = merge_crops_rgb(rgb_imgs, ms)
        merged_w = merge_crops_npy(w_imgs, ms)
        merged_w_rgb = channels_to_rgb(merged_w)

        fig, axes = plt.subplots(8, 1, figsize=(14, 11))

        axes[0].imshow(merged_rgb)
        axes[0].set_title("Original RGB merged crop", fontsize=10)
        axes[0].set_ylabel("RGB", fontsize=8)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(merged_w_rgb)
        axes[1].set_title("Weighted 6ch → colored overlay", fontsize=10)
        axes[1].set_ylabel("weighted", fontsize=8)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        for i, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
            ch = merged_w[:, :, i]
            px = int(ch.sum())
            axes[i+2].imshow(ch, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            axes[i+2].set_title(f"ch{i}: {name} ({px} px)", fontsize=9)
            axes[i+2].set_ylabel(name, fontsize=8)
            axes[i+2].set_xticks([]); axes[i+2].set_yticks([])

        fig.suptitle(f"Original vs Weighted: {lid} ({args.stars}★)", fontsize=12, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(PROJECT_ROOT, "models", f"weighted_vs_orig_{lid}.png")
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
