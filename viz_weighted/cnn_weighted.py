# 6-Channel Weighted CNN with Ordinal Regression
# Loads pre-rendered 6-channel .npy crops (with sparse types scaled up)
# No RGB→6ch conversion needed — data is already 6-channel binary.

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CROP_W = 128
CROP_H = 64
WINDOW = 3
PAD = 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)  # 386
N_CHANNELS = 6


# --------------- Dataset ---------------

class LevelDatasetWeighted(Dataset):
    """Loads 6-channel .npy crops. Pre-caches all to RAM on init."""

    def __init__(self, level_groups, crops_dir, preload=True):
        self.levels = level_groups
        self.crops_dir = crops_dir
        self.cache = {}

        if preload:
            all_files = set()
            for _, _, crop_infos in level_groups:
                for fn, _ in crop_infos:
                    all_files.add(fn)
            print(f"  Pre-loading {len(all_files)} .npy crops to RAM...", end=" ", flush=True)
            for fn in all_files:
                self.cache[fn] = np.load(os.path.join(crops_dir, fn))
            print("done.")

    def __len__(self):
        return len(self.levels)

    def _load(self, fn):
        try:
            if fn in self.cache:
                arr = self.cache[fn]
            else:
                arr = np.load(os.path.join(self.crops_dir, fn))
            if arr.shape != (CROP_H, CROP_W, N_CHANNELS):
                return np.zeros((CROP_H, CROP_W, N_CHANNELS), dtype=np.float32)
            return arr
        except Exception:
            return np.zeros((CROP_H, CROP_W, N_CHANNELS), dtype=np.float32)

    def __getitem__(self, idx):
        level_id, stars, crop_infos = self.levels[idx]
        crop_infos = sorted(crop_infos, key=lambda ci: ci[1])

        imgs = [self._load(fn) for fn, x in crop_infos]

        n = len(imgs)
        if n < WINDOW:
            while len(imgs) < WINDOW:
                imgs.append(np.zeros((CROP_H, CROP_W, N_CHANNELS), dtype=np.float32))
            n = WINDOW

        merged = []
        for i in range(n - WINDOW + 1):
            parts = []
            for j in range(WINDOW):
                if j > 0:
                    parts.append(np.zeros((CROP_H, PAD, N_CHANNELS), dtype=np.float32))
                parts.append(imgs[i + j])
            wide = np.concatenate(parts, axis=1)  # (H, MERGED_W, 6)
            wide = wide.transpose(2, 0, 1)         # (6, H, MERGED_W)
            merged.append(wide)

        merged = np.stack(merged, axis=0)
        label = int(stars) - 1
        return torch.from_numpy(merged), label


def collate_fn(batch):
    merged_list, labels = zip(*batch)
    max_m = max(m.shape[0] for m in merged_list)
    B = len(merged_list)
    C, H, W = merged_list[0].shape[1:]

    padded = torch.zeros(B, max_m, C, H, W)
    mask = torch.zeros(B, max_m, dtype=torch.bool)
    for i, m in enumerate(merged_list):
        n = m.shape[0]
        padded[i, :n] = m
        mask[i, :n] = True

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, mask


# --------------- Model ---------------

class MergedCropCNN6Ch(nn.Module):
    """
    6-channel input CNN with ordinal regression (CORAL).

    Dimension flow for (6, 64, 386) input:
      Conv1 + MaxPool(2): (16, 32, 193)
      Conv2 + MaxPool(2): (32, 16, 96)
      Conv3 + MaxPool(2): (64, 8, 48)
      AdaptiveAvgPool(4, 24): (64, 4, 24)
      Flatten: 6144
    """

    def __init__(self, num_classes=10, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 24)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 24, feat_dim),
            nn.ReLU(),
        )
        self.ordinal_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes - 1),
        )

    def forward(self, merged, mask):
        B, M, C, H, W = merged.shape
        x = merged.view(B * M, C, H, W)
        x = self.backbone(x)
        x = self.proj(x)
        x = x.view(B, M, -1)

        mask_f = mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.ordinal_head(x)

    def predict(self, merged, mask):
        logits = self.forward(merged, mask)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1)
        return preds


def ordinal_loss(logits, labels, num_classes=10):
    K = num_classes
    targets = torch.zeros_like(logits)
    for k in range(K - 1):
        targets[:, k] = (labels > k).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# --------------- Evaluation ---------------

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    off1 = 0
    total = 0
    mae_sum = 0.0
    with torch.no_grad():
        for merged, labels, mask in dataloader:
            merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
            preds = model.predict(merged, mask)
            correct += (preds == labels).sum().item()
            off1 += ((preds - labels).abs() <= 1).sum().item()
            mae_sum += (preds - labels).abs().float().sum().item()
            total += len(labels)
    return {
        "acc": correct / total,
        "off1": off1 / total,
        "mae": mae_sum / total,
        "n_levels": total,
    }


# --------------- Main ---------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    crops_dir = os.path.join(PROJECT_ROOT, "crops_weighted")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))
    print(f"Loaded {len(all_rows)} crops from manifest")

    level_map = defaultdict(lambda: {"stars": None, "crops": []})
    for r in all_rows:
        lid = r["level_id"]
        level_map[lid]["stars"] = r["stars"]
        level_map[lid]["crops"].append((r["filename"], int(r["x"])))

    level_groups = [(lid, d["stars"], d["crops"]) for lid, d in level_map.items()]
    print(f"Total levels: {len(level_groups)}")

    merged_counts = [max(1, len(d["crops"]) - WINDOW + 1) for d in level_map.values()]
    print(f"Merged crops per level: min={min(merged_counts)}, max={max(merged_counts)}, "
          f"total={sum(merged_counts)}")

    level_ids = sorted(level_map.keys())
    train_ids, test_ids = train_test_split(level_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    train_levels = [g for g in level_groups if g[0] in train_ids]
    val_levels = [g for g in level_groups if g[0] in val_ids]
    test_levels = [g for g in level_groups if g[0] in test_ids]
    print(f"Split: train={len(train_levels)}, val={len(val_levels)}, test={len(test_levels)} levels")

    print("Loading datasets...")
    train_ds = LevelDatasetWeighted(train_levels, crops_dir, preload=True)
    val_ds = LevelDatasetWeighted(val_levels, crops_dir, preload=True)
    test_ds = LevelDatasetWeighted(test_levels, crops_dir, preload=True)

    use_cuda = torch.cuda.is_available()
    dl_kwargs = dict(collate_fn=collate_fn,
                     num_workers=min(2, os.cpu_count() or 1),
                     pin_memory=use_cuda)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MergedCropCNN6Ch().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Loss: Ordinal regression (CORAL)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3)

    # Mixed precision (AMP) for CUDA — ~2x speedup on T4
    use_amp = use_cuda
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    if use_amp:
        print("Using mixed precision (AMP)")

    train_losses, val_losses, val_accs, val_off1s, val_maes = [], [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for merged, labels, mask in pbar:
            merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(merged, mask)
                loss = ordinal_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(labels)
            pbar.set_postfix(loss=f"{running_loss / ((pbar.n + 1) * args.batch_size):.4f}")
        train_loss = running_loss / len(train_ds)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for merged, labels, mask in val_dl:
                merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(merged, mask)
                    loss = ordinal_loss(logits, labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        metrics = evaluate(model, val_dl, device)
        val_accs.append(metrics["acc"])
        val_off1s.append(metrics["off1"])
        val_maes.append(metrics["mae"])

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {metrics['acc']:.4f} | Off1: {metrics['off1']:.4f} | "
              f"MAE: {metrics['mae']:.2f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(PROJECT_ROOT, "models", "cnn_weighted.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "cnn_weighted.pth"), weights_only=True))
    print("\n--- Final Results (best model) ---")
    for split_name, dl in [("Validation", val_dl), ("Test", test_dl)]:
        m = evaluate(model, dl, device)
        print(f"{split_name}: Acc={m['acc']:.4f}, Off1={m['off1']:.4f}, "
              f"MAE={m['mae']:.2f} ({m['n_levels']} levels)")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Ordinal Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(val_accs, label="Exact Acc")
    ax2.plot(val_off1s, label="Off-by-1 Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.plot(val_maes, label="MAE", color="red")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("MAE (stars)"); ax3.set_title("Mean Absolute Error")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "models", "cnn_weighted_curves.png"), dpi=150)
    plt.close()
    print("Saved training curves to models/cnn_weighted_curves.png")


if __name__ == "__main__":
    main()
