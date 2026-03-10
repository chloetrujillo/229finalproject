# Analyze how color-specific information (especially hazard=red) flows through CNN layers
# Compares random vs trained weights, shows per-channel activation maps

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CROP_W = 128
CROP_H = 64
WINDOW = 3
PAD = 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)  # 386

# Object group colors (from render_levels.py)
GROUP_COLORS = {
    "hazards":  np.array([230, 50, 50]) / 255.0,
    "blocks":   np.array([160, 160, 160]) / 255.0,
    "triggers": np.array([230, 210, 50]) / 255.0,
    "portals":  np.array([50, 220, 220]) / 255.0,
    "orbs":     np.array([50, 200, 80]) / 255.0,
    "unknown":  np.array([80, 80, 80]) / 255.0,
}


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
    return np.concatenate(parts, axis=1)


def create_object_mask(merged_np, color, threshold=0.1):
    """Create a binary mask for pixels matching a specific object color."""
    diff = np.abs(merged_np - color[np.newaxis, np.newaxis, :])
    return (diff.sum(axis=2) < threshold).astype(np.float32)


def compute_activation_at_locations(feat_map, mask, pool_factor):
    """Compute mean activation at object locations vs background.
    feat_map: (C, H, W) feature tensor
    mask: (H_orig, W_orig) binary mask at input resolution
    pool_factor: how much the feature map is downsampled
    Returns: (mean_at_object, mean_at_background) per channel
    """
    # Downsample mask to match feature map resolution
    from scipy.ndimage import zoom
    fh, fw = feat_map.shape[1], feat_map.shape[2]
    mask_resized = zoom(mask, (fh / mask.shape[0], fw / mask.shape[1]), order=0)
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    obj_count = mask_resized.sum()
    bg_count = (1 - mask_resized).sum()

    if obj_count < 1 or bg_count < 1:
        return None, None

    obj_activations = []
    bg_activations = []
    for c in range(feat_map.shape[0]):
        ch = feat_map[c]
        obj_mean = (ch * mask_resized).sum() / obj_count
        bg_mean = (ch * (1 - mask_resized)).sum() / bg_count
        obj_activations.append(obj_mean)
        bg_activations.append(bg_mean)

    return np.array(obj_activations), np.array(bg_activations)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1, help="Number of levels to analyze")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .pth")
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

    # Build CNN layers
    layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
    layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
    layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
    pool = nn.AdaptiveAvgPool2d((4, 8))
    layers = [layer1, layer2, layer3, pool]
    layer_names = ["Conv1 (32ch)", "Conv2 (64ch)", "Conv3 (128ch)", "AvgPool (4x8)"]

    mode_label = "random"
    if args.model:
        from src.model.cnn_multi_crop import MergedCropCNN
        model = MergedCropCNN()
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
        model.eval()
        backbone = model.backbone
        layers = [backbone[0:4], backbone[4:8], backbone[8:12], backbone[12]]
        mode_label = "trained"
        print(f"Loaded trained model from {args.model}")
    else:
        print("Using random weights")

    for layer in layers:
        if hasattr(layer, 'eval'):
            layer.eval()

    for lid in level_ids:
        rows = sorted(level_map[lid], key=lambda r: int(r["x"]))
        imgs = []
        for r in rows:
            img = Image.open(os.path.join(crops_dir, r["filename"]))
            img = np.array(img, dtype=np.float32) / 255.0
            imgs.append(img)

        merge_start = min(args.merge_idx, max(0, len(imgs) - WINDOW))
        merged_np = merge_crops(imgs, merge_start)
        merged_tensor = torch.from_numpy(merged_np.transpose(2, 0, 1)).unsqueeze(0)

        # Create object masks
        masks = {}
        for name, color in GROUP_COLORS.items():
            masks[name] = create_object_mask(merged_np, color)

        # Forward through layers
        with torch.no_grad():
            outputs = []
            x = merged_tensor
            for layer in layers:
                x = layer(x)
                outputs.append(x.squeeze(0).numpy())

        # ========== Figure 1: Color channel analysis ==========
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(5, 6, hspace=0.5, wspace=0.3)

        # Row 0: Original image + RGB channels + object masks
        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow((merged_np * 255).clip(0, 255).astype(np.uint8))
        ax.set_title("Merged crop (RGB)", fontsize=9)
        ax.axis("off")

        for ci, (cname, cmap) in enumerate([("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]):
            ax = fig.add_subplot(gs[0, ci + 2])
            ax.imshow(merged_np[:, :, ci], cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"{cname} channel", fontsize=8)
            ax.axis("off")

        # Show hazard mask
        ax = fig.add_subplot(gs[0, 5])
        mask_overlay = merged_np.copy()
        hazard_mask = masks["hazards"]
        mask_overlay[hazard_mask > 0.5] = [1, 0, 0]  # highlight hazards in pure red
        ax.imshow((mask_overlay * 255).clip(0, 255).astype(np.uint8))
        ax.set_title(f"Hazard pixels ({int(hazard_mask.sum())}px)", fontsize=8)
        ax.axis("off")

        # Rows 1-4: Feature maps with highest activation at hazard locations
        pool_factors = [2, 4, 8, 1]  # cumulative downsampling
        for li, (feat, name) in enumerate(zip(outputs, layer_names)):
            # Compute activation strength at hazard vs background locations
            obj_act, bg_act = compute_activation_at_locations(feat, hazard_mask, pool_factors[li])
            if obj_act is None:
                continue

            # Find channels most responsive to hazards (highest obj/bg ratio)
            ratio = obj_act / (bg_act + 1e-8)
            top_channels = np.argsort(ratio)[::-1][:6]

            for j, ch_idx in enumerate(top_channels):
                ax = fig.add_subplot(gs[li + 1, j])
                ax.imshow(feat[ch_idx], cmap="hot", aspect="auto")
                r = ratio[ch_idx]
                ax.set_title(f"ch{ch_idx} (h/bg={r:.1f})", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(name, fontsize=8)

        fig.suptitle(f"Color Analysis: {lid} ({level_stars[lid]}★) [{mode_label} weights]\n"
                     f"Top 6 channels most responsive to hazard (red) locations",
                     fontsize=11, y=1.02)
        out1 = os.path.join(PROJECT_ROOT, "models", f"color_analysis_{mode_label}_{lid}.png")
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out1}")

        # ========== Figure 2: Activation strength by object type ==========
        fig2, axes2 = plt.subplots(1, len(layers), figsize=(16, 4))
        object_types = ["hazards", "blocks", "triggers", "portals", "orbs"]
        colors = ["red", "gray", "gold", "cyan", "green"]

        for li, (feat, name) in enumerate(zip(outputs, layer_names)):
            ax = axes2[li]
            means = []
            labels = []
            for oi, otype in enumerate(object_types):
                m = masks[otype]
                if m.sum() < 5:
                    continue
                obj_act, bg_act = compute_activation_at_locations(feat, m, pool_factors[li])
                if obj_act is None:
                    continue
                # Mean activation across all channels at object locations
                mean_obj = obj_act.mean()
                mean_bg = bg_act.mean()
                means.append(mean_obj)
                labels.append(otype)

            if means:
                x_pos = range(len(means))
                bar_colors = [colors[object_types.index(l)] for l in labels]
                ax.bar(x_pos, means, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel("Mean activation", fontsize=8)
            ax.grid(True, alpha=0.3)

        fig2.suptitle(f"Mean activation by object type: {lid} ({level_stars[lid]}★) [{mode_label}]",
                      fontsize=11)
        plt.tight_layout()
        out2 = os.path.join(PROJECT_ROOT, "models", f"activation_by_type_{mode_label}_{lid}.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out2}")

        # ========== Print summary stats ==========
        print(f"\n--- Object pixel counts in merged crop ---")
        for otype in object_types:
            count = int(masks[otype].sum())
            pct = count / (MERGED_W * CROP_H) * 100
            print(f"  {otype:12s}: {count:5d} px ({pct:5.1f}%)")

        print(f"\n--- Hazard activation ratio (object/background) per layer ---")
        for li, (feat, name) in enumerate(zip(outputs, layer_names)):
            obj_act, bg_act = compute_activation_at_locations(feat, masks["hazards"], pool_factors[li])
            if obj_act is not None:
                ratio = obj_act / (bg_act + 1e-8)
                print(f"  {name:20s}: mean ratio={ratio.mean():.2f}, "
                      f"max={ratio.max():.2f} (ch {ratio.argmax()}), "
                      f"channels with ratio>2: {(ratio > 2).sum()}/{len(ratio)}")


if __name__ == "__main__":
    main()
