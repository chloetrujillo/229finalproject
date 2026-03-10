# Visualize merged crops and CNN feature maps at each layer
# Supports both random weights and trained model weights

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CROP_W = 128
CROP_H = 64
WINDOW = 3
PAD = 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)  # 386


def merge_crops(imgs, start_idx=0):
    """Merge 3 crops horizontally with 1px black padding."""
    parts = []
    for j in range(WINDOW):
        if j > 0:
            parts.append(np.zeros((CROP_H, PAD, 3), dtype=np.float32))
        idx = start_idx + j
        if idx < len(imgs):
            parts.append(imgs[idx])
        else:
            parts.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.float32))
    return np.concatenate(parts, axis=1)  # (64, 386, 3)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=2, help="Number of levels to visualize")
    parser.add_argument("--channels", type=int, default=8, help="Feature map channels to show per layer")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pth (optional)")
    parser.add_argument("--merge-idx", type=int, default=0, help="Which merged group to visualize (default: 0)")
    args = parser.parse_args()

    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    # Group crops by level
    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))

    level_map = defaultdict(list)
    level_stars = {}
    for r in all_rows:
        if r["stars"] == str(args.stars):
            level_map[r["level_id"]].append(r)
            level_stars[r["level_id"]] = r["stars"]

    level_ids = sorted(level_map.keys())[:args.limit]

    # Build CNN layers (matching MergedCropCNN backbone)
    layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
    layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
    layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
    pool = nn.AdaptiveAvgPool2d((4, 8))
    layers = [layer1, layer2, layer3, pool]
    layer_names = [
        "Conv1 (32ch, 32×193)",
        "Conv2 (64ch, 16×96)",
        "Conv3 (128ch, 8×48)",
        "AvgPool (128ch, 4×8)",
    ]

    # Load trained weights if provided
    if args.model:
        from src.model.cnn_multi_crop import MergedCropCNN
        model = MergedCropCNN()
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
        model.eval()
        # Extract backbone layers
        backbone = model.backbone
        layers = [backbone[0:4], backbone[4:8], backbone[8:12], backbone[12]]
        print(f"Loaded trained model from {args.model}")
    else:
        print("Using random weights (no model loaded)")

    for layer in layers:
        if hasattr(layer, 'eval'):
            layer.eval()

    for lid in level_ids:
        rows = sorted(level_map[lid], key=lambda r: int(r["x"]))

        # Load crop images
        imgs = []
        for r in rows:
            img = Image.open(os.path.join(crops_dir, r["filename"]))
            img = np.array(img, dtype=np.float32) / 255.0
            imgs.append(img)

        # Merge crops
        merge_start = min(args.merge_idx, max(0, len(imgs) - WINDOW))
        merged_np = merge_crops(imgs, merge_start)  # (64, 386, 3)
        merged_tensor = torch.from_numpy(merged_np.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, 64, 386)

        # Forward through layers
        with torch.no_grad():
            outputs = []
            x = merged_tensor
            for layer in layers:
                x = layer(x)
                outputs.append(x)

        # Plot
        n_layers = len(layers)
        n_ch = min(args.channels, 8)
        fig, axes = plt.subplots(n_layers + 1, n_ch, figsize=(n_ch * 2.5, (n_layers + 1) * 1.8))

        # Row 0: merged crop image
        merged_display = (merged_np * 255).clip(0, 255).astype(np.uint8)
        for j in range(n_ch):
            if j == 0:
                axes[0, j].imshow(merged_display)
                axes[0, j].set_title("Merged crop", fontsize=9)
            else:
                axes[0, j].axis("off")
        axes[0, 0].set_ylabel(f"Input\n{CROP_H}×{MERGED_W}", fontsize=8)

        # Rows 1+: feature maps
        for i, (feat, name) in enumerate(zip(outputs, layer_names)):
            feat_np = feat.squeeze(0).numpy()  # (C, H, W)
            n_total = feat_np.shape[0]
            indices = np.linspace(0, n_total - 1, n_ch, dtype=int)
            for j, ch_idx in enumerate(indices):
                ax = axes[i + 1, j]
                ax.imshow(feat_np[ch_idx], cmap="viridis", aspect="auto")
                if j == 0:
                    ax.set_ylabel(name, fontsize=7)
                ax.set_title(f"ch {ch_idx}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

        crop_ids = [rows[merge_start + k]["sample_idx"] if merge_start + k < len(rows) else "pad"
                    for k in range(WINDOW)]
        fig.suptitle(f"Level {lid} ({level_stars[lid]}★) - merged crops [{','.join(crop_ids)}]",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(PROJECT_ROOT, "models", f"feature_vis_merged_{lid}.png")
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
