"""What did SCHP call the collar region under the chin, and is the final pale there?"""
from pathlib import Path
import numpy as np
from PIL import Image
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.domains.portrait import human_parsing as hp
labels = np.asarray(Image.open(ROOT / "evidence/p29-collar-repair/labels_fresh.png"))
orig = np.asarray(Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
final = np.asarray(Image.open(ROOT / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
names = {i: n for i, n in enumerate(hp.ATR_CLASSES)}
print("ATR classes:", names)
print("CLOTHING_CLASSES:", hp.CLOTHING_CLASSES, [names[c] for c in hp.CLOTHING_CLASSES])
for (y0,y1,x0,x1,label) in [(594,760,430,620,"collar band all"),(600,700,470,560,"under-chin strip"),(600,700,560,640,"collar right")]:
    sub = labels[y0:y1, x0:x1]
    u, c = np.unique(sub, return_counts=True)
    tot = sub.size
    info = ", ".join(f"{names.get(int(k),k)}={v}({v/tot:.0%})" for k, v in zip(u, c) if v/tot > 0.01)
    print(f"{label}: {info}")
# block label map in the strip
print("label ids grid (x 440..600 step 20, y 590..750 step 20):")
for cy in range(590, 751, 20):
    print(f"y{cy:4d} " + " ".join(f"{int(labels[cy,cx]):2d}" for cx in range(440, 601, 20)))
# how pale is the final in the strip?
for (y0,y1,x0,x1,label) in [(594,760,430,620,"strip"),(600,700,470,560,"under-chin")]:
    f = final[y0:y1,x0:x1].reshape(-1,3)
    o = orig[y0:y1,x0:x1].reshape(-1,3)
    fl = 0.299*f[:,0]+0.587*f[:,1]+0.114*f[:,2]
    ol = 0.299*o[:,0]+0.587*o[:,1]+0.114*o[:,2]
    print(f"{label}: final luma mean {fl.mean():.1f} p50 {np.median(fl):.1f} | orig luma mean {ol.mean():.1f} | final pale(>150) share {np.mean(fl>150):.2f} orig pale share {np.mean(ol>150):.2f}")
