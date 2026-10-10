"""Coarse HSV grid over the under-chin collar region of the best final."""
from pathlib import Path
import numpy as np
from PIL import Image
import colorsys
ROOT = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(ROOT / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)

def block(cx, cy, s=10):
    sub = final[cy-s:cy+s, cx-s:cx+s].reshape(-1,3)/255.0
    mx=sub.max(1); mn=sub.min(1); d=mx-mn
    sat=np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0)
    val=mx
    r,g,b=sub[:,0],sub[:,1],sub[:,2]
    hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return np.median(hue), np.median(sat), np.median(val)

print("     " + "".join(f"{x:>16d}" for x in range(420, 601, 30)))
for cy in range(570, 781, 30):
    row=[]
    for cx in range(420, 601, 30):
        h,s,v = block(cx,cy)
        row.append(f"H{h:5.0f} S{s:4.2f} V{v:4.2f}")
    print(f"y{cy:4d} " + "".join(row))
