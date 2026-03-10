# Merged-crop CNN for predicting GD level difficulty
# Sliding window: every 3 adjacent crops (sorted by x) merged into one wide image

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

CROP_W = 128
CROP_H = 64
WINDOW = 3
PAD = 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)  # 386


# --------------- Dataset ---------------

class LevelDataset(Dataset):
    """Each item returns merged crops (sliding window of 3) for one level."""

    def __init__(self, level_groups, crops_dir):
        # level_groups: list of (level_id, stars, [(filename, x), ...])
        self.levels = level_groups
        self.crops_dir = crops_dir

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        level_id, stars, crop_infos = self.levels[idx]
        # Sort by x coordinate
        crop_infos = sorted(crop_infos, key=lambda ci: ci[1])

        # Load all crop images
        imgs = []
        for fn, x in crop_infos:
            img = Image.open(os.path.join(self.crops_dir, fn))
            img = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
            imgs.append(img)

        # Sliding window merge: window=3, stride=1
        n = len(imgs)
        merged = []
        if n < WINDOW:
            # Pad with black images to fill window
            while len(imgs) < WINDOW:
                imgs.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.float32))
            n = WINDOW

        for i in range(n - WINDOW + 1):
            parts = []
            for j in range(WINDOW):
                if j > 0:
                    parts.append(np.zeros((CROP_H, PAD, 3), dtype=np.float32))
                parts.append(imgs[i + j])
            wide = np.concatenate(parts, axis=1)  # (H, MERGED_W, 3)
            wide = wide.transpose(2, 0, 1)  # (3, H, MERGED_W)
            merged.append(wide)

        merged = np.stack(merged, axis=0)  # (M, 3, H, MERGED_W)
        label = int(stars) - 1
        return torch.from_numpy(merged), label


def collate_fn(batch):
    """Pad variable-length merged crops to max M in batch, return mask."""
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

class MergedCropCNN(nn.Module):
    def __init__(self, num_classes=10, feat_dim=256):
        super().__init__()
        # CNN backbone for 386×64 merged crops
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, feat_dim),
            nn.ReLU(),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, merged, mask):
        """
        merged: (B, M, 3, H, W)  - M merged crops per level
        mask:   (B, M) bool
        Returns: logits (B, num_classes)
        """
        B, M, C, H, W = merged.shape
        x = merged.view(B * M, C, H, W)
        x = self.backbone(x)
        x = self.proj(x)                    # (B*M, feat_dim)
        x = x.view(B, M, -1)               # (B, M, feat_dim)

        # Mean pooling with mask
        mask_f = mask.unsqueeze(-1).float()  # (B, M, 1)
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, feat_dim)

        return self.classifier(x)


# --------------- Evaluation ---------------

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    off1 = 0
    total = 0
    with torch.no_grad():
        for merged, labels, mask in dataloader:
            merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
            logits = model(merged, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            off1 += ((preds - labels).abs() <= 1).sum().item()
            total += len(labels)
    return {
        "acc": correct / total,
        "off1": off1 / total,
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

    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    # Load manifest and group by level, keeping x coordinate
    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))
    print(f"Loaded {len(all_rows)} crops from manifest")

    level_map = defaultdict(lambda: {"stars": None, "crops": []})
    for r in all_rows:
        lid = r["level_id"]
        level_map[lid]["stars"] = r["stars"]
        level_map[lid]["crops"].append((r["filename"], int(r["x"])))

    # level_groups: (level_id, stars, [(filename, x), ...])
    level_groups = [(lid, d["stars"], d["crops"]) for lid, d in level_map.items()]
    print(f"Total levels: {len(level_groups)}")

    # Stats
    merged_counts = [max(1, len(d["crops"]) - WINDOW + 1) for d in level_map.values()]
    print(f"Merged crops per level: min={min(merged_counts)}, max={max(merged_counts)}, "
          f"total={sum(merged_counts)}")

    # Split by level_id (same split as cnn_model.py)
    level_ids = sorted(level_map.keys())
    train_ids, test_ids = train_test_split(level_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    train_levels = [g for g in level_groups if g[0] in train_ids]
    val_levels = [g for g in level_groups if g[0] in val_ids]
    test_levels = [g for g in level_groups if g[0] in test_ids]
    print(f"Split: train={len(train_levels)}, val={len(val_levels)}, test={len(test_levels)} levels")

    train_ds = LevelDataset(train_levels, crops_dir)
    val_ds = LevelDataset(val_levels, crops_dir)
    test_ds = LevelDataset(test_levels, crops_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MergedCropCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses, val_losses, val_accs, val_off1s = [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for merged, labels, mask in pbar:
            merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(merged, mask), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            pbar.set_postfix(loss=f"{running_loss / ((pbar.n + 1) * args.batch_size):.4f}")
        train_loss = running_loss / len(train_ds)
        train_losses.append(train_loss)

        # Validate
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for merged, labels, mask in val_dl:
                merged, labels, mask = merged.to(device), labels.to(device), mask.to(device)
                loss = criterion(model(merged, mask), labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        metrics = evaluate(model, val_dl, device)
        val_accs.append(metrics["acc"])
        val_off1s.append(metrics["off1"])

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Level Acc: {metrics['acc']:.4f} | Level Off1: {metrics['off1']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(PROJECT_ROOT, "models", "cnn_multi_crop.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "cnn_multi_crop.pth"), weights_only=True))
    print("\n--- Final Results (best model) ---")
    for split_name, dl in [("Validation", val_dl), ("Test", test_dl)]:
        m = evaluate(model, dl, device)
        print(f"{split_name}: Level Acc={m['acc']:.4f}, Level Off1={m['off1']:.4f} "
              f"({m['n_levels']} levels)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(val_accs, label="Exact Acc")
    ax2.plot(val_off1s, label="Off-by-1 Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "models", "cnn_multi_crop_curves.png"), dpi=150)
    plt.close()
    print("Saved training curves to models/cnn_multi_crop_curves.png")


if __name__ == "__main__":
    main()
