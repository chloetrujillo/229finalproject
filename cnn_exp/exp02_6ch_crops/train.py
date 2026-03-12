#!/usr/bin/env python3
"""Experiment 02 - Training: 6-Channel Crops CNN with Ordinal Regression
Converts RGB crops to 6 binary channels at load time, merges 3 adjacent crops
via sliding window, and uses CORAL ordinal regression.
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CROP_W, CROP_H, WINDOW, PAD = 128, 64, 3, 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)  # 386
N_CH = 6
CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
GROUP_COLORS_U8 = {
    "hazards": np.array([230,50,50], dtype=np.uint8),
    "blocks": np.array([160,160,160], dtype=np.uint8),
    "triggers": np.array([230,210,50], dtype=np.uint8),
    "portals": np.array([50,220,220], dtype=np.uint8),
    "orbs": np.array([50,200,80], dtype=np.uint8),
    "unknown": np.array([80,80,80], dtype=np.uint8),
}

def rgb_to_6ch(img_u8):
    h, w = img_u8.shape[:2]
    out = np.zeros((h, w, N_CH), dtype=np.float32)
    for i, name in enumerate(CHANNEL_NAMES):
        out[:, :, i] = np.all(img_u8 == GROUP_COLORS_U8[name], axis=2).astype(np.float32)
    return out


class LevelDataset6Ch(Dataset):
    def __init__(self, level_groups, crops_dir, preload=True):
        self.levels = level_groups
        self.crops_dir = crops_dir
        self.cache = {}
        if preload:
            all_files = {fn for _, _, ci in level_groups for fn, _ in ci}
            print(f"  Pre-loading {len(all_files)} crops...", end=" ", flush=True)
            for fn in all_files:
                self.cache[fn] = rgb_to_6ch(np.array(Image.open(os.path.join(crops_dir, fn))))
            print("done.")

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        _, stars, crop_infos = self.levels[idx]
        crop_infos = sorted(crop_infos, key=lambda ci: ci[1])
        imgs = [self.cache.get(fn, rgb_to_6ch(np.array(Image.open(os.path.join(self.crops_dir, fn)))))
                for fn, _ in crop_infos]
        n = len(imgs)
        while len(imgs) < WINDOW:
            imgs.append(np.zeros((CROP_H, CROP_W, N_CH), dtype=np.float32))
        n = max(n, WINDOW)
        merged = []
        for i in range(n - WINDOW + 1):
            parts = []
            for j in range(WINDOW):
                if j > 0: parts.append(np.zeros((CROP_H, PAD, N_CH), dtype=np.float32))
                parts.append(imgs[i + j])
            merged.append(np.concatenate(parts, axis=1).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(merged)), int(stars) - 1


def collate_fn(batch):
    merged_list, labels = zip(*batch)
    max_m = max(m.shape[0] for m in merged_list)
    B, C, H, W = len(merged_list), *merged_list[0].shape[1:]
    padded = torch.zeros(B, max_m, C, H, W)
    mask = torch.zeros(B, max_m, dtype=torch.bool)
    for i, m in enumerate(merged_list):
        padded[i, :m.shape[0]] = m; mask[i, :m.shape[0]] = True
    return padded, torch.tensor(labels, dtype=torch.long), mask


class MergedCropCNN6Ch(nn.Module):
    def __init__(self, K=10, feat=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(N_CH,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,8)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128*4*8, feat), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, K-1))

    def forward(self, merged, mask):
        B, M, C, H, W = merged.shape
        x = self.proj(self.backbone(merged.view(B*M, C, H, W))).view(B, M, -1)
        mf = mask.unsqueeze(-1).float()
        return self.head((x * mf).sum(1) / mf.sum(1).clamp(min=1))

    def predict(self, merged, mask):
        return (torch.sigmoid(self.forward(merged, mask)) > 0.5).sum(dim=1)


def ordinal_loss(logits, labels, K=10):
    targets = torch.zeros_like(logits)
    for k in range(K - 1): targets[:, k] = (labels > k).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def evaluate(model, dl, device):
    model.eval()
    correct = off1 = total = 0; mae = 0.0
    with torch.no_grad():
        for m, l, mask in dl:
            m, l, mask = m.to(device), l.to(device), mask.to(device)
            p = model.predict(m, mask)
            correct += (p == l).sum().item()
            off1 += ((p-l).abs() <= 1).sum().item()
            mae += (p-l).abs().float().sum().item()
            total += len(l)
    return {"acc": correct/total, "off1": off1/total, "mae": mae/total, "n": total}


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    crops_dir = os.path.join(PROJECT_ROOT, "data", "crops")
    with open(os.path.join(crops_dir, "manifest.csv")) as f:
        all_rows = list(csv.DictReader(f))

    lm = defaultdict(lambda: {"stars": None, "crops": []})
    for r in all_rows:
        lm[r["level_id"]]["stars"] = r["stars"]
        lm[r["level_id"]]["crops"].append((r["filename"], int(r["x"])))
    groups = [(lid, d["stars"], d["crops"]) for lid, d in lm.items()]

    ids = sorted(lm.keys())
    tr, te = train_test_split(ids, test_size=0.2, random_state=42)
    va, te = train_test_split(te, test_size=0.5, random_state=42)
    tr, va, te = set(tr), set(va), set(te)

    trl = [g for g in groups if g[0] in tr]
    val = [g for g in groups if g[0] in va]
    tel = [g for g in groups if g[0] in te]
    print(f"Split: train={len(trl)}, val={len(val)}, test={len(tel)}")

    trds = LevelDataset6Ch(trl, crops_dir); vds = LevelDataset6Ch(val, crops_dir); teds = LevelDataset6Ch(tel, crops_dir)
    kw = dict(collate_fn=collate_fn, num_workers=2)
    trdl = DataLoader(trds, batch_size=args.batch_size, shuffle=True, **kw)
    vdl = DataLoader(vds, batch_size=args.batch_size, **kw)
    tedl = DataLoader(teds, batch_size=args.batch_size, **kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MergedCropCNN6Ch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best, pc = float("inf"), 0
    md = os.path.join(PROJECT_ROOT, "models")
    for ep in range(args.epochs):
        model.train(); rl = 0.0
        for m, l, mask in tqdm(trdl, desc=f"Epoch {ep+1}", leave=False):
            m, l, mask = m.to(device), l.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = ordinal_loss(model(m, mask), l)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            rl += loss.item() * len(l)
        vl = 0.0; model.eval()
        with torch.no_grad():
            for m, l, mask in vdl:
                m, l, mask = m.to(device), l.to(device), mask.to(device)
                vl += ordinal_loss(model(m, mask), l).item() * len(l)
        vl /= len(val)
        met = evaluate(model, vdl, device)
        print(f"Epoch {ep+1} | ValLoss={vl:.4f} Acc={met['acc']:.4f} Off1={met['off1']:.4f} MAE={met['mae']:.2f}")
        if vl < best:
            best, pc = vl, 0
            os.makedirs(md, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(md, "cnn_6ch_crops.pth"))
        else:
            pc += 1
            if pc >= args.patience: print(f"Early stop ep {ep+1}"); break

    model.load_state_dict(torch.load(os.path.join(md, "cnn_6ch_crops.pth"), weights_only=True))
    for n, dl in [("Val", vdl), ("Test", tedl)]:
        m = evaluate(model, dl, device)
        print(f"{n}: Acc={m['acc']:.4f} Off1={m['off1']:.4f} MAE={m['mae']:.2f}")

if __name__ == "__main__":
    main()
