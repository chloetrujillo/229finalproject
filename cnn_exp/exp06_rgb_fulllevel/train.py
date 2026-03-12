#!/usr/bin/env python3
"""Experiment 06 - Training: RGB Full-Level CNN
One resized image (64x512x3) per level. 4-layer CNN + CORAL ordinal regression.
"""
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H, W, NCH = 64, 512, 3

class FullDS(Dataset):
    def __init__(self, samples, dd, aug=False, preload=True):
        self.samples, self.dd, self.aug, self.cache = samples, dd, aug, {}
        if preload:
            print(f"  Loading {len(samples)} levels...", end=" ", flush=True)
            for _, _, fp in samples:
                try: self.cache[fp] = np.load(os.path.join(dd, fp))
                except: self.cache[fp] = np.zeros((H, W, NCH), dtype=np.float32)
            print("done.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        _, stars, fp = self.samples[idx]
        a = self.cache.get(fp)
        if a is None:
            try: a = np.load(os.path.join(self.dd, fp))
            except: a = np.zeros((H, W, NCH), dtype=np.float32)
        if a.shape != (H, W, NCH): a = np.zeros((H, W, NCH), dtype=np.float32)
        if a.dtype == np.uint8: a = a.astype(np.float32) / 255.0
        if self.aug and np.random.random() > 0.5: a = a[:, ::-1, :].copy()
        return torch.from_numpy(a.transpose(2, 0, 1)), int(stars) - 1

class FullCNN(nn.Module):
    def __init__(self, K=10, feat=256):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(NCH, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 8)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(256 * 2 * 8, feat), nn.ReLU(), nn.Dropout(0.4))
        self.head = nn.Linear(feat, K - 1)
    def forward(self, x): return self.head(self.proj(self.bb(x)))
    def predict(self, x): return (torch.sigmoid(self.forward(x)) > 0.5).sum(dim=1)

def ordinal_loss(lo, la, K=10):
    t = torch.zeros_like(lo)
    for k in range(K - 1): t[:, k] = (la > k).float()
    return nn.functional.binary_cross_entropy_with_logits(lo, t)

def evaluate(model, dl, dev):
    model.eval(); c = o = t = 0; mae = 0.0
    with torch.no_grad():
        for im, la in dl:
            im, la = im.to(dev), la.to(dev); p = model.predict(im)
            c += (p == la).sum().item(); o += ((p - la).abs() <= 1).sum().item()
            mae += (p - la).abs().float().sum().item(); t += len(la)
    return {"acc": c / t, "off1": o / t, "mae": mae / t, "n": t}

def main():
    pa = ArgumentParser()
    pa.add_argument("--epochs", type=int, default=30); pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--lr", type=float, default=1e-3); pa.add_argument("--patience", type=int, default=5)
    pa.add_argument("--no-augment", action="store_true")
    a = pa.parse_args(); torch.manual_seed(42); np.random.seed(42)
    dd = os.path.join(PROJECT_ROOT, "data", "levels_resized_rgb")
    samples = []
    for n in range(1, 11):
        sd = os.path.join(dd, f"{n}stars")
        if not os.path.isdir(sd): continue
        for fn in os.listdir(sd):
            if fn.endswith(".npy"): samples.append((fn.replace(".npy", ""), n, os.path.join(f"{n}stars", fn)))
    print(f"Total: {len(samples)} levels")
    ids = [s[0] for s in samples]
    tr, te = train_test_split(ids, test_size=0.2, random_state=42)
    va, te = train_test_split(te, test_size=0.5, random_state=42)
    tr, va, te = set(tr), set(va), set(te)
    aug = not a.no_augment
    tds = FullDS([s for s in samples if s[0] in tr], dd, aug=aug)
    vds = FullDS([s for s in samples if s[0] in va], dd)
    teds = FullDS([s for s in samples if s[0] in te], dd)
    kw = dict(num_workers=2, pin_memory=torch.cuda.is_available())
    trdl = DataLoader(tds, batch_size=a.batch_size, shuffle=True, **kw)
    vdl = DataLoader(vds, batch_size=a.batch_size, **kw)
    tedl = DataLoader(teds, batch_size=a.batch_size, **kw)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = FullCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=3)
    amp = torch.cuda.is_available(); scaler = torch.amp.GradScaler(enabled=amp)
    best, pc, md = float("inf"), 0, os.path.join(PROJECT_ROOT, "models")
    for ep in range(a.epochs):
        model.train(); rl = 0
        for im, la in tqdm(trdl, desc=f"Ep {ep + 1}", leave=False):
            im, la = im.to(dev), la.to(dev); opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=amp): loss = ordinal_loss(model(im), la)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item() * len(la)
        vl = 0; model.eval()
        with torch.no_grad():
            for im, la in vdl:
                im, la = im.to(dev), la.to(dev)
                vl += ordinal_loss(model(im), la).item() * len(la)
        vl /= len(vds); sched.step(vl)
        met = evaluate(model, vdl, dev)
        print(f"Ep {ep + 1} VL={vl:.4f} Acc={met['acc']:.4f} Off1={met['off1']:.4f} MAE={met['mae']:.2f}")
        if vl < best: best, pc = vl, 0; os.makedirs(md, exist_ok=True); torch.save(model.state_dict(), os.path.join(md, "cnn_rgb_full.pth"))
        else:
            pc += 1
            if pc >= a.patience: print("Early stop"); break
    model.load_state_dict(torch.load(os.path.join(md, "cnn_rgb_full.pth"), weights_only=True))
    for n, dl in [("Val", vdl), ("Test", tedl)]:
        m = evaluate(model, dl, dev); print(f"{n}: Acc={m['acc']:.4f} Off1={m['off1']:.4f} MAE={m['mae']:.2f}")

if __name__ == "__main__": main()
