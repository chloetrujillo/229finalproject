#!/usr/bin/env python3
"""Experiment 04 - Data Preparation: 6-Channel Weighted Crops
Step 1: Render levels as 6-channel images with per-type stamp scaling
Step 2: Sample fixed-size crops from 6ch images
"""
import json, os, csv, random, numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CATEGORY_TO_GROUP = {"block":"blocks","spikes":"hazards","saws":"hazards","fire":"hazards","breakable":"hazards","trigger_color":"triggers","trigger_move":"triggers","trigger_pulse":"triggers","trigger_alpha":"triggers","trigger_spawn":"triggers","trigger_rotate":"triggers","trigger_follow":"triggers","trigger_shake":"triggers","trigger_toggle":"triggers","trigger_transition":"triggers","trigger_startpos":"triggers","trigger_other":"triggers","portal_gamemode":"portals","portal_speed":"portals","portal_size":"portals","portal_mirror":"portals","portal_dual":"portals","orb":"orbs","pulsing":"orbs","coin":"orbs","glow":"skip","fading":"skip","particles":"skip","clouds":"skip","arrows":"skip","text":"skip","pixels":"skip","hands":"skip","monsters":"skip","pickups":"skip","collisions":"skip","unknown":"unknown"}
CHANNEL_NAMES = ["hazards", "blocks", "triggers", "portals", "orbs", "unknown"]
CHANNEL_INDEX = {n: i for i, n in enumerate(CHANNEL_NAMES)}
TYPE_WEIGHT = {"hazards": 3, "blocks": 1, "triggers": 1, "portals": 3, "orbs": 3, "unknown": 1}
UNIT = 30

def render_weighted(ld, idc, scale=2):
    objs = ld.get("data", [])
    if not objs: return None
    pts = {g: [] for g in CHANNEL_NAMES}
    for o in objs:
        x,y,oid = o.get("x"),o.get("y"),o.get("id")
        if x is None or y is None: continue
        c = idc.get(str(oid),"unknown") if oid else "unknown"
        g = CATEGORY_TO_GROUP.get(c,"unknown")
        if g != "skip": pts[g].append((x,y))
    ax = [p[0] for v in pts.values() for p in v]
    if not ax: return None
    ay = [p[1] for v in pts.values() for p in v]
    xn,xx,yn,yx = min(ax),max(ax),min(ay),max(ay)
    w,h = int((xx-xn)/UNIT)+1, int((yx-yn)/UNIT)+1
    ms = scale * max(TYPE_WEIGHT.values()); pad = ms // 2
    iw, ih = w*scale+2*pad, h*scale+2*pad
    if iw*ih > 2e8: return None
    img = np.zeros((ih, iw, 6), dtype=np.float32)
    for g in CHANNEL_NAMES:
        ci, wt, ss = CHANNEL_INDEX[g], TYPE_WEIGHT[g], scale*TYPE_WEIGHT[g]
        for px,py in pts[g]:
            ix,iy = int((px-xn)/UNIT), int((yx-py)/UNIT)
            cx,cy = pad+ix*scale+scale//2, pad+iy*scale+scale//2
            x0,y0 = max(0,cx-ss//2), max(0,cy-ss//2)
            img[y0:min(ih,y0+ss), x0:min(iw,x0+ss), ci] = 1.0
    ac = img.sum(axis=2) > 0
    rows,cols = np.any(ac,axis=1), np.any(ac,axis=0)
    if rows.any() and cols.any():
        r0,r1 = np.where(rows)[0][[0,-1]]; c0,c1 = np.where(cols)[0][[0,-1]]
        img = img[r0:r1+1, c0:c1+1]
    return img

def sample_6ch(img, n, cw, ch, k, retries=50):
    h,w = img.shape[:2]
    ph,pw = max(0,ch-h), max(0,cw-w)
    if ph or pw: img = np.pad(img,((0,ph),(0,pw),(0,0)),mode='constant'); h,w = img.shape[:2]
    crops = []
    for _ in range(n):
        for _ in range(retries):
            x,y = random.randint(0,w-cw), random.randint(0,h-ch)
            crop = img[y:y+ch, x:x+cw]
            nz = int((crop.sum(axis=2)>0).sum())
            if nz >= k: crops.append((crop,x,y,nz)); break
    return crops

def main():
    p = ArgumentParser()
    p.add_argument("--scale",type=int,default=2); p.add_argument("--n",type=int,default=10)
    p.add_argument("--crop-w",type=int,default=128); p.add_argument("--crop-h",type=int,default=64)
    p.add_argument("--k",type=int,default=40); p.add_argument("--stars",type=int,default=None)
    p.add_argument("--limit",type=int,default=None); p.add_argument("--seed",type=int,default=42)
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    with open(os.path.join(PROJECT_ROOT,"src","data","id_to_category.json")) as f: idc = json.load(f)
    sr = [a.stars] if a.stars else range(1,11)
    crops_dir = os.path.join(PROJECT_ROOT,"data","crops_weighted"); manifest = []
    for ns in sr:
        dd = os.path.join(PROJECT_ROOT,"levels",f"{ns}stars")
        pd = os.path.join(PROJECT_ROOT,"data","plots_weighted",f"{ns}stars")
        cd = os.path.join(crops_dir,f"{ns}stars")
        if not os.path.isdir(dd): continue
        os.makedirs(pd,exist_ok=True); os.makedirs(cd,exist_ok=True)
        files = sorted(f for f in os.listdir(dd) if f.endswith(".json"))
        if a.limit: files = files[:a.limit]
        for fn in tqdm(files,desc=f"{ns}stars"):
            lid = fn.split(".")[0]
            pp = os.path.join(pd,f"{lid}.npy")
            if not os.path.exists(pp):
                with open(os.path.join(dd,fn)) as f: ld = json.load(f)
                img = render_weighted(ld,idc,scale=a.scale)
                if img is None: continue
                np.save(pp,img)
            else: img = np.load(pp)
            for i,(crop,x,y,nz) in enumerate(sample_6ch(img,a.n,a.crop_w,a.crop_h,a.k)):
                on = f"{lid}_{i}.npy"
                np.save(os.path.join(cd,on),crop)
                manifest.append({"filename":f"{ns}stars/{on}","stars":ns,"level_id":lid,"sample_idx":i,"x":x,"y":y,"nonzero_pixels":nz})
    mp = os.path.join(crops_dir,"manifest.csv")
    with open(mp,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["filename","stars","level_id","sample_idx","x","y","nonzero_pixels"])
        w.writeheader(); w.writerows(manifest)
    print(f"Done. {len(manifest)} crops")

if __name__ == "__main__": main()
