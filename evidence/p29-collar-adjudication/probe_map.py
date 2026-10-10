from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
def LCs(a):
    a=a.astype(np.float32)
    L=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    C=a.max(2)-a.min(2)
    return L,C
orig=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
o=orig.astype(np.int32)/255.0
mx=o.max(2);mn=o.min(2);d=mx-mn
r,g,b=o[...,0],o[...,1],o[...,2]
hue=np.zeros(mx.shape);m=d>1e-6
i=(mx==r)&m;hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
i=(mx==g)&m;hue[i]=60*((b[i]-r[i])/d[i])+120
i=(mx==b)&m;hue[i]=60*((r[i]-g[i])/d[i])+240
oL,oC=LCs(orig);fL,fC=LCs(final)
xs=list(range(428,606,8)); ys=list(range(572,766,8))
hdr="     "+"".join(f"{x:>5d}" for x in xs)
print("=== FINAL (L,C) ===");print(hdr)
for y in ys:
    row=f"{y:>4d} "
    for x in xs:
        row+=f"{int(fL[y,x]):>3d}/{int(fC[y,x]):<2d}"
    print(row)
print("=== ORIG (L,C,H) ===");print(hdr)
for y in ys:
    row=f"{y:>4d} "
    for x in xs:
        row+=f"{int(oL[y,x]):>3d},{int(oC[y,x]):>2d},{int(hue[y,x]):>3d} "
    print(row)
print("=== MASK (R=release g=guarded .=neither) ===");print(hdr)
for y in ys:
    row=f"{y:>4d} "
    for x in xs:
        row+=f"  {'R' if rel[y,x] else ('g' if gp[y,x] else '.')}  "
    print(row)
