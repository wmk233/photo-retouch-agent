from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
def lc(a):
    l = 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    c = a.max(2)-a.min(2)
    return l, c
fl, fc = lc(final); ol, oc = lc(orig)
for y in range(600, 720, 15):
    row=[]
    for x in range(440, 600, 20):
        row.append(f"({x},{y}) F[l={fl[y,x]:.0f},c={fc[y,x]:.0f}] O[l={ol[y,x]:.0f},c={oc[y,x]:.0f}]")
    print(" | ".join(row))
