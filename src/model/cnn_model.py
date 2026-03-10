# CNN model for predicting GD level difficulty from image crops

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


class CropDataset(Dataset):
    def __init__(self, rows, crops_dir):
        self.rows = rows
        self.crops_dir = crops_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(os.path.join(self.crops_dir, row["filename"]))
        img = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
        img = img.transpose(2, 0, 1)  # CHW
        label = int(row["stars"]) - 1  # 0-9
        return torch.from_numpy(img), label, row["level_id"]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, dataloader, device):
    """Evaluate model, return crop-level and level-level metrics."""
    model.eval()
    crop_correct = 0
    crop_off1 = 0
    crop_total = 0

    level_probs = defaultdict(list)
    level_labels = {}

    with torch.no_grad():
        for imgs, labels, level_ids in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            crop_correct += (preds == labels).sum().item()
            crop_off1 += ((preds - labels).abs() <= 1).sum().item()
            crop_total += len(labels)

            for i in range(len(level_ids)):
                lid = level_ids[i]
                level_probs[lid].append(probs[i].cpu().numpy())
                level_labels[lid] = labels[i].item()

    # Level-level: average softmax -> argmax
    level_correct = 0
    level_off1 = 0
    for lid, prob_list in level_probs.items():
        avg_prob = np.mean(prob_list, axis=0)
        pred = avg_prob.argmax()
        true = level_labels[lid]
        if pred == true:
            level_correct += 1
        if abs(pred - true) <= 1:
            level_off1 += 1

    n_levels = len(level_probs)
    return {
        "crop_acc": crop_correct / crop_total,
        "crop_off1": crop_off1 / crop_total,
        "level_acc": level_correct / n_levels,
        "level_off1": level_off1 / n_levels,
        "n_crops": crop_total,
        "n_levels": n_levels,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    crops_dir = os.path.join(PROJECT_ROOT, "crops")
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    # Load manifest
    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))
    print(f"Loaded {len(all_rows)} crops from manifest")

    # Split by level_id
    level_ids = sorted(set(r["level_id"] for r in all_rows))
    train_ids, test_ids = train_test_split(level_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    train_rows = [r for r in all_rows if r["level_id"] in train_ids]
    val_rows = [r for r in all_rows if r["level_id"] in val_ids]
    test_rows = [r for r in all_rows if r["level_id"] in test_ids]
    print(f"Split: train={len(train_rows)} crops ({len(train_ids)} levels), "
          f"val={len(val_rows)} crops ({len(val_ids)} levels), "
          f"test={len(test_rows)} crops ({len(test_ids)} levels)")

    train_ds = CropDataset(train_rows, crops_dir)
    val_ds = CropDataset(val_rows, crops_dir)
    test_ds = CropDataset(test_rows, crops_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    val_accs = []
    val_off1s = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
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
            for imgs, labels, _ in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = criterion(model(imgs), labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        metrics = evaluate(model, val_dl, device)
        val_accs.append(metrics["crop_acc"])
        val_off1s.append(metrics["crop_off1"])

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Crop Acc: {metrics['crop_acc']:.4f} | Crop Off1: {metrics['crop_off1']:.4f} | "
              f"Level Acc: {metrics['level_acc']:.4f} | Level Off1: {metrics['level_off1']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "models", "cnn_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models", "cnn_model.pth"),
                                     weights_only=True))
    print("\n--- Final Results (best model) ---")
    for split_name, dl in [("Validation", val_dl), ("Test", test_dl)]:
        m = evaluate(model, dl, device)
        print(f"{split_name}: "
              f"Crop Acc={m['crop_acc']:.4f}, Crop Off1={m['crop_off1']:.4f} ({m['n_crops']} crops) | "
              f"Level Acc={m['level_acc']:.4f}, Level Off1={m['level_off1']:.4f} ({m['n_levels']} levels)")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accs, label="Exact Acc")
    ax2.plot(val_off1s, label="Off-by-1 Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "models", "cnn_train_curves.png"), dpi=150)
    plt.close()
    print(f"Saved training curves to models/cnn_train_curves.png")


if __name__ == "__main__":
    main()
