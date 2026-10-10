"""Sample tones: white-knit residual vs the new olive blouse, in the best final."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
FINAL = np.asarray(Image.open(ROOT / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
ORIG = np.asarray(Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)

def stat(name, y0, y1, x0, x1, img=FINAL):
    sub = img[y0:y1, x0:x1].reshape(-1, 3)
    luma = 0.299*sub[:,0]+0.587*sub[:,1]+0.114*sub[:,2]
    chroma = sub.max(1)-sub.min(1)
    print(f"{name:28s} n={len(sub):7d} mean={sub.mean(0).round(1)} luma={luma.mean():6.1f} chroma={chroma.mean():5.1f} p90chroma={np.percentile(chroma,90):5.1f}")

# residual white knit patch under the chin (from the overlay)
stat("final: residual under chin L", 610, 700, 380, 500)
stat("final: residual under chin R", 620, 700, 520, 590)
stat("final: blouse light ribbon", 900, 1000, 620, 720)
stat("final: blouse body lower", 1050, 1150, 300, 430)
stat("final: blouse body right", 1000, 1100, 700, 800)
stat("final: neck skin", 560, 590, 430, 520)
print("---- original (white knit sweater) ----")
stat("orig: knit under chin", 610, 700, 380, 500, ORIG)
stat("orig: knit chest", 950, 1050, 400, 600, ORIG)
print("---- neutral-ness of blouse vs knit ----")
for name,(y0,y1,x0,x1),img in [("final blouse",(1000,1100,700,800),FINAL),("final residual",(610,700,380,500),FINAL),("orig knit",(950,1050,400,600),ORIG),("final neck skin",(560,590,430,520),FINAL)]:
    sub = img[y0:y1, x0:x1].reshape(-1,3)
    gb = sub[:,1]-sub[:,2]
    print(f"{name:16s} G-B mean={gb.mean():6.1f} p10={np.percentile(gb,10):6.1f} share(G-B<20)={np.mean(gb<20):.3f}")
