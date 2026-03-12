#!/usr/bin/env python3
"""Experiment 01 - Training: RGB Crops CNN
3-channel RGB crops (128x64) with CrossEntropy loss.
Evaluates both crop-level and level-level accuracy.
"""

import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class CropDataset(Dataset):
    def __init__(self, rows, crops_dir):
        self.rows = rows
        self.crops_dir = crops_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = np.array(Image.open(os.path.join(self.crops_dir, row["filename"])),
                       dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW
        label = int(row["stars"]) - 1
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
        return self.classifier(self.features(x))


def evaluate(model, dataloader, device):
    model.eval()
    crop_correct = crop_off1 = crop_total = 0
    level_probs, level_labels = defaultdict(list), {}
    with torch.no_grad():
        for imgs, labels, level_ids in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            crop_correct += (preds == labels).sum().item()
            crop_off1 += ((preds - labels).abs() <= 1).sum().item()
            crop_total += len(labels)
            for i in range(len(level_ids)):
                level_probs[level_ids[i]].append(probs[i].cpu().numpy())
                level_labels[level_ids[i]] = labels[i].item()

    level_correct = level_off1 = 0
    for lid, prob_list in level_probs.items():
        pred = np.mean(prob_list, axis=0).argmax()
        true = level_labels[lid]
        level_correct += int(pred == true)
        level_off1 += int(abs(pred - true) <= 1)

    n = len(level_probs)
    return {"crop_acc": crop_correct/crop_total, "crop_off1": crop_off1/crop_total,
            "level_acc": level_correct/n, "level_off1": level_off1/n,
            "n_crops": crop_total, "n_levels": n}


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)

    crops_dir = os.path.join(PROJECT_ROOT, "data", "crops")
    with open(os.path.join(crops_dir, "manifest.csv")) as f:
        all_rows = list(csv.DictReader(f))
    print(f"Loaded {len(all_rows)} crops")

    level_ids = sorted(set(r["level_id"] for r in all_rows))
    train_ids, test_ids = train_test_split(level_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    train_rows = [r for r in all_rows if r["level_id"] in train_ids]
    val_rows = [r for r in all_rows if r["level_id"] in val_ids]
    test_rows = [r for r in all_rows if r["level_id"] in test_ids]
    print(f"Split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} crops")

    train_dl = DataLoader(CropDataset(train_rows, crops_dir), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(CropDataset(val_rows, crops_dir), batch_size=args.batch_size, num_workers=2)
    test_dl = DataLoader(CropDataset(test_rows, crops_dir), batch_size=args.batch_size, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss, patience_counter = float("inf"), 0
    models_dir = os.path.join(PROJECT_ROOT, "models")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, labels, _ in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(labels)
        train_losses.append(running / len(train_rows))

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels, _ in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss += criterion(model(imgs), labels).item() * len(labels)
        val_losses.append(val_loss / len(val_rows))

        m = evaluate(model, val_dl, device)
        val_accs.append(m["level_acc"])
        print(f"Epoch {epoch+1} | TrainLoss={train_losses[-1]:.4f} ValLoss={val_losses[-1]:.4f} "
              f"CropAcc={m['crop_acc']:.4f} LevelAcc={m['level_acc']:.4f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(models_dir, "cnn_rgb_crops.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(models_dir, "cnn_rgb_crops.pth"), weights_only=True))
    print("\n--- Final Results ---")
    for name, dl in [("Val", val_dl), ("Test", test_dl)]:
        m = evaluate(model, dl, device)
        print(f"{name}: CropAcc={m['crop_acc']:.4f} LevelAcc={m['level_acc']:.4f} "
              f"LevelOff1={m['level_off1']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(val_accs, label="Level Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "exp01_curves.png"), dpi=150)
    print("Saved training curves")


if __name__ == "__main__":
    main()
