#!/usr/bin/env python3
"""Experiment 03 - Data Preparation: Multi-Crop RGB CNN
Same pipeline as exp01 — renders RGB PNGs then samples crops.
The sliding window merge of 3 crops happens at training time.
"""
# Identical to exp01/prepare_data.py. See that file for full documentation.
# Run: python prepare_data.py  (or reuse exp01's data/)

import json, os, csv, random, numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CATEGORY_TO_GROUP = {"block":"blocks","spikes":"hazards","saws":"hazards","fire":"hazards","breakable":"hazards","trigger_color":"triggers","trigger_move":"triggers","trigger_pulse":"triggers","trigger_alpha":"triggers","trigger_spawn":"triggers","trigger_rotate":"triggers","trigger_follow":"triggers","trigger_shake":"triggers","trigger_toggle":"triggers","trigger_transition":"triggers","trigger_startpos":"triggers","trigger_other":"triggers","portal_gamemode":"portals","portal_speed":"portals","portal_size":"portals","portal_mirror":"portals","portal_dual":"portals","orb":"orbs","pulsing":"orbs","coin":"orbs","glow":"skip","fading":"skip","particles":"skip","clouds":"skip","arrows":"skip","text":"skip","pixels":"skip","hands":"skip","monsters":"skip","pickups":"skip","collisions":"skip","unknown":"unknown"}
GROUP_COLORS = {"unknown":(80,80,80),"blocks":(160,160,160),"hazards":(230,50,50),"triggers":(230,210,50),"portals":(50,220,220),"orbs":(50,200,80)}
DRAW_ORDER = ["unknown","blocks","triggers","orbs","hazards","portals"]
UNIT = 30

def render_level(ld, idc, scale=1):
    objs = ld.get("data", [])
    if not objs: return None
    pts = {g: [] for g in DRAW_ORDER}
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
    iw,ih = w*scale, h*scale
    if iw*ih > 2e8: return None
    img = np.zeros((ih,iw,3), dtype=np.uint8)
    for g in DRAW_ORDER:
        cl = GROUP_COLORS[g]
        for px,py in pts[g]:
            ix,iy = int((px-xn)/UNIT), int((yx-py)/UNIT)
            x0,y0 = ix*scale, iy*scale
            img[y0:min(y0+scale,ih), x0:min(x0+scale,iw)] = cl
    return Image.fromarray(img)

def sample_crops(ia, n, cw, ch, k, retries=50):
    h,w = ia.shape[:2]
    ph,pw = max(0,ch-h), max(0,cw-w)
    if ph or pw: ia = np.pad(ia,((0,ph),(0,pw),(0,0)),mode='constant'); h,w = ia.shape[:2]
    crops = []
    for _ in range(n):
        for _ in range(retries):
            x,y = random.randint(0,w-cw), random.randint(0,h-ch)
            crop = ia[y:y+ch, x:x+cw]
            nb = int(np.sum(crop.sum(axis=2)>0))
            if nb >= k: crops.append((crop,x,y,nb)); break
    return crops

def main():
    p = ArgumentParser()
    p.add_argument("--scale",type=int,default=1); p.add_argument("--n",type=int,default=5)
    p.add_argument("--crop-w",type=int,default=128); p.add_argument("--crop-h",type=int,default=64)
    p.add_argument("--k",type=int,default=20); p.add_argument("--stars",type=int,default=None)
    p.add_argument("--limit",type=int,default=None); p.add_argument("--seed",type=int,default=42)
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    with open(os.path.join(PROJECT_ROOT,"src","data","id_to_category.json")) as f: idc = json.load(f)
    sr = [a.stars] if a.stars else range(1,11)
    cd = os.path.join(PROJECT_ROOT,"data","crops"); manifest = []
    for ns in sr:
        dd = os.path.join(PROJECT_ROOT,"levels",f"{ns}stars")
        pd = os.path.join(PROJECT_ROOT,"data","plots",f"{ns}stars")
        crd = os.path.join(cd,f"{ns}stars")
        if not os.path.isdir(dd): continue
        os.makedirs(pd,exist_ok=True); os.makedirs(crd,exist_ok=True)
        files = sorted(f for f in os.listdir(dd) if f.endswith(".json"))
        if a.limit: files = files[:a.limit]
        for fn in tqdm(files,desc=f"{ns}stars"):
            lid = fn.split(".")[0]; pp = os.path.join(pd,f"{lid}.png")
            if not os.path.exists(pp):
                with open(os.path.join(dd,fn)) as f: ld = json.load(f)
                img = render_level(ld,idc,scale=a.scale)
                if img is None: continue
                img.save(pp)
            else: img = Image.open(pp)
            for i,(crop,x,y,nb) in enumerate(sample_crops(np.array(img),a.n,a.crop_w,a.crop_h,a.k)):
                on = f"{lid}_{i}.png"
                Image.fromarray(crop).save(os.path.join(crd,on))
                manifest.append({"filename":f"{ns}stars/{on}","stars":ns,"level_id":lid,"sample_idx":i,"x":x,"y":y,"nonblack_pixels":nb})
    mp = os.path.join(cd,"manifest.csv")
    with open(mp,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["filename","stars","level_id","sample_idx","x","y","nonblack_pixels"])
        w.writeheader(); w.writerows(manifest)
    print(f"Done. {len(manifest)} crops")

if __name__ == "__main__": main()
