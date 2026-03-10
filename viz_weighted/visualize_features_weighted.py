# Visualize feature maps at each CNN layer for the weighted 6-channel model
# Shows how the 6-channel input is transformed through Conv1→Conv2→Conv3→AvgPool

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from cnn_weighted import MergedCropCNN6Ch, LevelDatasetWeighted, collate_fn, N_CHANNELS, CROP_H, CROP_W, WINDOW, PAD

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
CHANNEL_COLORS = [
    [1.0, 0.2, 0.2],    # hazards - red
    [0.63, 0.63, 0.63],  # blocks - grey
    [0.9, 0.82, 0.2],    # triggers - yellow
    [0.2, 0.86, 0.86],   # portals - cyan
    [0.2, 0.78, 0.31],   # orbs - green
    [0.31, 0.31, 0.31],  # unknown - dark grey
]


def channels_to_rgb(img_6ch):
    """Convert (6, H, W) tensor to RGB for display."""
    if isinstance(img_6ch, torch.Tensor):
        img_6ch = img_6ch.cpu().numpy()
    h, w = img_6ch.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    draw_order = [5, 1, 2, 4, 0, 3]
    for idx in draw_order:
        color = np.array(CHANNEL_COLORS[idx])
        mask = img_6ch[idx] > 0.5
        rgb[mask] = color
    return rgb


def get_layer_outputs(model, x):
    """Extract feature maps after each conv block and avgpool."""
    outputs = {}
    layers = list(model.backbone)

    # backbone is: Conv,BN,ReLU,MaxPool, Conv,BN,ReLU,MaxPool, Conv,BN,ReLU,MaxPool, AdaptiveAvgPool
    block_ends = [3, 7, 11, 12]  # indices after MaxPool1, MaxPool2, MaxPool3, AvgPool
    block_names = ["Conv1+Pool", "Conv2+Pool", "Conv3+Pool", "AvgPool"]

    cur = x
    for i, layer in enumerate(layers):
        cur = layer(cur)
        if i in block_ends:
            name = block_names[block_ends.index(i)]
            outputs[name] = cur.detach().cpu()

    return outputs


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .pth file")
    parser.add_argument("--stars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8, help="Top-k feature maps to show per layer")
    args = parser.parse_args()

    crops_dir = os.path.join(PROJECT_ROOT, "crops_weighted")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))

    level_map = defaultdict(lambda: {"stars": None, "crops": []})
    for r in all_rows:
        lid = r["level_id"]
        level_map[lid]["stars"] = r["stars"]
        level_map[lid]["crops"].append((r["filename"], int(r["x"])))

    # Pick levels with requested star rating
    target_levels = [(lid, d["stars"], d["crops"])
                     for lid, d in level_map.items()
                     if int(d["stars"]) == args.stars][:args.limit]

    if not target_levels:
        print(f"No levels found with {args.stars} stars")
        return

    device = torch.device("cpu")

    if args.model:
        # Auto-detect model size from checkpoint
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        ch1 = ckpt["backbone.0.weight"].shape[0]  # first conv out channels
        feat = ckpt["proj.1.weight"].shape[0]
        if ch1 == 32:
            model = MergedCropCNN6Ch(feat_dim=feat)  # old large model
            # Temporarily restore large backbone
            import torch.nn as nn
            model.backbone = nn.Sequential(
                nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 8)),
            )
            model.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 8, feat), nn.ReLU())
            model.ordinal_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 9))
        else:
            model = MergedCropCNN6Ch()
        model.load_state_dict(ckpt)
        print(f"Loaded model from {args.model} (conv1_ch={ch1}, feat={feat})")
    else:
        model = MergedCropCNN6Ch()
        print("No --model specified, using random weights")

    model.eval()

    ds = LevelDatasetWeighted(target_levels, crops_dir, preload=True)

    out_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(len(ds)):
        merged_tensor, label = ds[idx]
        level_id = target_levels[idx][0]

        # Use the first merged crop (window)
        single = merged_tensor[0:1]  # (1, 6, H, W)

        # Get layer outputs
        with torch.no_grad():
            layer_outputs = get_layer_outputs(model, single)

        # --- Plot ---
        n_layers = len(layer_outputs)
        rows = 1 + n_layers  # input + each layer
        fig, axes = plt.subplots(rows, args.top_k, figsize=(args.top_k * 2.5, rows * 2))

        # Row 0: input channels (show up to top_k of the 6 channels)
        input_img = single[0]  # (6, H, W)
        for j in range(min(args.top_k, 6)):
            ax = axes[0][j]
            ax.imshow(input_img[j].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"in: {CHANNEL_NAMES[j]}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        # Fill remaining cols with colored overlay
        if args.top_k > 6:
            ax = axes[0][6]
            ax.imshow(channels_to_rgb(input_img))
            ax.set_title("overlay", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for j in range(7, args.top_k):
                axes[0][j].axis("off")
        axes[0][0].set_ylabel("Input", fontsize=9, rotation=0, labelpad=40, va="center")

        # Rows 1+: feature maps per layer (top-k by activation strength)
        for row_i, (layer_name, feat) in enumerate(layer_outputs.items(), start=1):
            feat = feat[0]  # (C, H, W)
            n_ch = feat.shape[0]

            # Rank channels by total activation
            ch_sums = feat.view(n_ch, -1).sum(dim=1)
            top_indices = ch_sums.argsort(descending=True)[:args.top_k]

            for j, ch_idx in enumerate(top_indices):
                ax = axes[row_i][j]
                fm = feat[ch_idx].numpy()
                ax.imshow(fm, cmap="viridis", aspect="auto")
                ax.set_title(f"ch{ch_idx.item()} ({fm.max():.1f})", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            for j in range(len(top_indices), args.top_k):
                axes[row_i][j].axis("off")

            size_str = "x".join(str(s) for s in feat.shape)
            axes[row_i][0].set_ylabel(f"{layer_name}\n{size_str}", fontsize=8,
                                       rotation=0, labelpad=50, va="center")

        title = f"Feature Maps: {level_id} ({args.stars}★)"
        if args.model:
            title += " [trained]"
        else:
            title += " [random]"
        fig.suptitle(title, fontsize=12, y=1.01)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"features_weighted_{level_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
