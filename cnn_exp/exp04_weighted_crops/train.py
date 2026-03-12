#!/usr/bin/env python3
"""Experiment 04 - Training: 6-Channel Weighted Crops CNN (BEST MODEL)
Pre-rendered 6ch .npy crops with per-type stamp scaling.
CORAL ordinal regression. ReduceLROnPlateau. AMP.
"""
import os, csv, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CW, CH, WIN, PAD, NCH = 128, 64, 3, 1, 6

class LevelDS(Dataset):
    def __init__(self, levels, cd, preload=True):
        self.levels, self.cd, self.cache = levels, cd, {}
        if preload:
            af = {fn for _,_,ci in levels for fn,_ in ci}
            print(f"  Loading {len(af)} crops...", end=" ", flush=True)
            for fn in af:
                try: self.cache[fn] = np.load(os.path.join(cd, fn))
                except: self.cache[fn] = np.zeros((CH,CW,NCH), dtype=np.float32)
            print("done.")
    def __len__(self): return len(self.levels)
    def _load(self, fn):
        a = self.cache.get(fn)
        if a is None:
            try: a = np.load(os.path.join(self.cd, fn))
            except: a = np.zeros((CH,CW,NCH), dtype=np.float32)
        return a if a.shape == (CH,CW,NCH) else np.zeros((CH,CW,NCH), dtype=np.float32)
    def __getitem__(self, idx):
        _,stars,ci = self.levels[idx]
        ci = sorted(ci, key=lambda c: c[1])
        imgs = [self._load(fn) for fn,_ in ci]
        while len(imgs) < WIN: imgs.append(np.zeros((CH,CW,NCH), dtype=np.float32))
        merged = []
        for i in range(len(imgs)-WIN+1):
            parts = []
            for j in range(WIN):
                if j: parts.append(np.zeros((CH,PAD,NCH), dtype=np.float32))
                parts.append(imgs[i+j])
            merged.append(np.concatenate(parts,axis=1).transpose(2,0,1))
        return torch.from_numpy(np.stack(merged)), int(stars)-1

def collate_fn(batch):
    ml,la = zip(*batch)
    mm = max(m.shape[0] for m in ml)
    B,C,H,W = len(ml),*ml[0].shape[1:]
    p = torch.zeros(B,mm,C,H,W); mask = torch.zeros(B,mm,dtype=torch.bool)
    for i,m in enumerate(ml): p[i,:m.shape[0]]=m; mask[i,:m.shape[0]]=True
    return p, torch.tensor(la,dtype=torch.long), mask

class CNN6Ch(nn.Module):
    def __init__(self, K=10, feat=128):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(NCH,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,24)))
        self.proj = nn.Sequential(nn.Flatten(),nn.Linear(64*4*24,feat),nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3),nn.Linear(feat,K-1))
    def forward(self,m,mask):
        B,M,C,H,W = m.shape
        x = self.proj(self.bb(m.view(B*M,C,H,W))).view(B,M,-1)
        mf = mask.unsqueeze(-1).float()
        return self.head((x*mf).sum(1)/mf.sum(1).clamp(min=1))
    def predict(self,m,mask):
        return (torch.sigmoid(self.forward(m,mask))>0.5).sum(dim=1)

def ordinal_loss(logits,labels,K=10):
    t = torch.zeros_like(logits)
    for k in range(K-1): t[:,k] = (labels>k).float()
    return nn.functional.binary_cross_entropy_with_logits(logits,t)

def evaluate(model,dl,dev):
    model.eval(); c=o=t=0; mae=0.0
    with torch.no_grad():
        for m,l,mask in dl:
            m,l,mask = m.to(dev),l.to(dev),mask.to(dev)
            p = model.predict(m,mask)
            c+=(p==l).sum().item(); o+=((p-l).abs()<=1).sum().item()
            mae+=(p-l).abs().float().sum().item(); t+=len(l)
    return {"acc":c/t,"off1":o/t,"mae":mae/t,"n":t}

def main():
    pa = ArgumentParser()
    pa.add_argument("--epochs",type=int,default=30); pa.add_argument("--batch-size",type=int,default=16)
    pa.add_argument("--lr",type=float,default=1e-3); pa.add_argument("--patience",type=int,default=5)
    a = pa.parse_args(); torch.manual_seed(42); np.random.seed(42)

    cd = os.path.join(PROJECT_ROOT,"data","crops_weighted")
    with open(os.path.join(cd,"manifest.csv")) as f: rows = list(csv.DictReader(f))
    lm = defaultdict(lambda:{"stars":None,"crops":[]})
    for r in rows: lm[r["level_id"]]["stars"]=r["stars"]; lm[r["level_id"]]["crops"].append((r["filename"],int(r["x"])))
    groups = [(lid,d["stars"],d["crops"]) for lid,d in lm.items()]
    ids = sorted(lm.keys())
    tr,te = train_test_split(ids,test_size=0.2,random_state=42)
    va,te = train_test_split(te,test_size=0.5,random_state=42)
    tr,va,te = set(tr),set(va),set(te)

    kw = dict(collate_fn=collate_fn,num_workers=2)
    trdl = DataLoader(LevelDS([g for g in groups if g[0] in tr],cd),batch_size=a.batch_size,shuffle=True,**kw)
    vdl = DataLoader(LevelDS([g for g in groups if g[0] in va],cd),batch_size=a.batch_size,**kw)
    tedl = DataLoader(LevelDS([g for g in groups if g[0] in te],cd),batch_size=a.batch_size,**kw)

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN6Ch().to(dev)
    opt = torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.3,patience=3)
    amp = torch.cuda.is_available(); scaler = torch.amp.GradScaler(enabled=amp)
    best,pc,md = float("inf"),0,os.path.join(PROJECT_ROOT,"models")

    for ep in range(a.epochs):
        model.train(); rl=0
        for m,l,mask in tqdm(trdl,desc=f"Ep {ep+1}",leave=False):
            m,l,mask = m.to(dev),l.to(dev),mask.to(dev)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda",enabled=amp): loss = ordinal_loss(model(m,mask),l)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item()*len(l)
        vl = 0; model.eval()
        with torch.no_grad():
            for m,l,mask in vdl:
                m,l,mask = m.to(dev),l.to(dev),mask.to(dev)
                vl += ordinal_loss(model(m,mask),l).item()*len(l)
        vl /= len(va); sched.step(vl)
        met = evaluate(model,vdl,dev)
        print(f"Ep {ep+1} VL={vl:.4f} Acc={met['acc']:.4f} Off1={met['off1']:.4f} MAE={met['mae']:.2f}")
        if vl < best: best,pc=vl,0; os.makedirs(md,exist_ok=True); torch.save(model.state_dict(),os.path.join(md,"cnn_weighted.pth"))
        else:
            pc+=1
            if pc>=a.patience: print("Early stop"); break
    model.load_state_dict(torch.load(os.path.join(md,"cnn_weighted.pth"),weights_only=True))
    for n,dl in [("Val",vdl),("Test",tedl)]:
        m = evaluate(model,dl,dev); print(f"{n}: Acc={m['acc']:.4f} Off1={m['off1']:.4f} MAE={m['mae']:.2f}")

if __name__ == "__main__": main()
