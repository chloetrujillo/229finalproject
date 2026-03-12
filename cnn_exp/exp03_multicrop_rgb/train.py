#!/usr/bin/env python3
"""Experiment 03 - Training: Multi-Crop RGB CNN
Merges 3 adjacent RGB crops into a wide (3, 64, 386) image via sliding window.
Uses CrossEntropyLoss with mask-based mean pooling across merged crops.
"""
import os, csv, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CROP_W, CROP_H, WINDOW, PAD = 128, 64, 3, 1
MERGED_W = CROP_W * WINDOW + PAD * (WINDOW - 1)

class LevelDataset(Dataset):
    def __init__(self, levels, crops_dir):
        self.levels, self.crops_dir = levels, crops_dir
    def __len__(self): return len(self.levels)
    def __getitem__(self, idx):
        _, stars, ci = self.levels[idx]
        ci = sorted(ci, key=lambda c: c[1])
        imgs = [np.array(Image.open(os.path.join(self.crops_dir, fn)), dtype=np.float32)/255 for fn, _ in ci]
        while len(imgs) < WINDOW: imgs.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.float32))
        n = len(imgs)
        merged = []
        for i in range(n - WINDOW + 1):
            parts = []
            for j in range(WINDOW):
                if j > 0: parts.append(np.zeros((CROP_H, PAD, 3), dtype=np.float32))
                parts.append(imgs[i+j])
            merged.append(np.concatenate(parts, axis=1).transpose(2,0,1))
        return torch.from_numpy(np.stack(merged)), int(stars) - 1

def collate_fn(batch):
    ml, labels = zip(*batch)
    mm = max(m.shape[0] for m in ml)
    B, C, H, W = len(ml), *ml[0].shape[1:]
    padded = torch.zeros(B, mm, C, H, W)
    mask = torch.zeros(B, mm, dtype=torch.bool)
    for i, m in enumerate(ml): padded[i,:m.shape[0]] = m; mask[i,:m.shape[0]] = True
    return padded, torch.tensor(labels, dtype=torch.long), mask

class MergedCropCNN(nn.Module):
    def __init__(self, K=10, feat=256):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,8)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128*4*8, feat), nn.ReLU())
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, K))

    def forward(self, m, mask):
        B, M, C, H, W = m.shape
        x = self.proj(self.bb(m.view(B*M, C, H, W))).view(B, M, -1)
        mf = mask.unsqueeze(-1).float()
        return self.cls((x * mf).sum(1) / mf.sum(1).clamp(min=1))

def evaluate(model, dl, device):
    model.eval(); c = o = t = 0
    with torch.no_grad():
        for m, l, mask in dl:
            m, l, mask = m.to(device), l.to(device), mask.to(device)
            p = model(m, mask).argmax(1)
            c += (p==l).sum().item(); o += ((p-l).abs()<=1).sum().item(); t += len(l)
    return {"acc": c/t, "off1": o/t, "n": t}

def main():
    pa = ArgumentParser()
    pa.add_argument("--epochs",type=int,default=30); pa.add_argument("--batch-size",type=int,default=16)
    pa.add_argument("--lr",type=float,default=1e-3); pa.add_argument("--patience",type=int,default=5)
    a = pa.parse_args(); torch.manual_seed(42); np.random.seed(42)

    cd = os.path.join(PROJECT_ROOT, "data", "crops")
    with open(os.path.join(cd, "manifest.csv")) as f: rows = list(csv.DictReader(f))
    lm = defaultdict(lambda: {"stars": None, "crops": []})
    for r in rows: lm[r["level_id"]]["stars"]=r["stars"]; lm[r["level_id"]]["crops"].append((r["filename"],int(r["x"])))
    groups = [(lid, d["stars"], d["crops"]) for lid, d in lm.items()]

    ids = sorted(lm.keys())
    tr, te = train_test_split(ids, test_size=0.2, random_state=42)
    va, te = train_test_split(te, test_size=0.5, random_state=42)
    tr, va, te = set(tr), set(va), set(te)

    kw = dict(collate_fn=collate_fn, num_workers=2)
    trdl = DataLoader(LevelDataset([g for g in groups if g[0] in tr], cd), batch_size=a.batch_size, shuffle=True, **kw)
    vdl = DataLoader(LevelDataset([g for g in groups if g[0] in va], cd), batch_size=a.batch_size, **kw)
    tedl = DataLoader(LevelDataset([g for g in groups if g[0] in te], cd), batch_size=a.batch_size, **kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MergedCropCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, pc, md = float("inf"), 0, os.path.join(PROJECT_ROOT, "models")

    for ep in range(a.epochs):
        model.train(); rl = 0
        for m, l, mask in tqdm(trdl, desc=f"Ep {ep+1}", leave=False):
            m,l,mask = m.to(device),l.to(device),mask.to(device)
            optimizer.zero_grad(); loss = criterion(model(m,mask),l); loss.backward(); optimizer.step()
            rl += loss.item()*len(l)
        vl = 0; model.eval()
        with torch.no_grad():
            for m,l,mask in vdl: m,l,mask=m.to(device),l.to(device),mask.to(device); vl+=criterion(model(m,mask),l).item()*len(l)
        met = evaluate(model, vdl, device)
        print(f"Ep {ep+1} ValLoss={vl/len(va):.4f} Acc={met['acc']:.4f} Off1={met['off1']:.4f}")
        if vl < best: best,pc=vl,0; os.makedirs(md,exist_ok=True); torch.save(model.state_dict(),os.path.join(md,"cnn_multicrop.pth"))
        else:
            pc += 1
            if pc >= a.patience: print(f"Early stop"); break
    model.load_state_dict(torch.load(os.path.join(md,"cnn_multicrop.pth"),weights_only=True))
    for n,dl in [("Val",vdl),("Test",tedl)]:
        m = evaluate(model,dl,device); print(f"{n}: Acc={m['acc']:.4f} Off1={m['off1']:.4f}")

if __name__ == "__main__": main()
