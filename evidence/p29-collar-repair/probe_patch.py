"""Measure HSV of the candidate residual patch vs blouse, in the best final."""
from pathlib import Path
import numpy as np
from PIL import Image
import colorsys
ROOT = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(ROOT / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
orig  = np.asarray(Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)

def hsv_stats(name, y0,y1,x0,x1, img=final):
    sub = img[y0:y1, x0:x1].reshape(-1,3)/255.0
    mx = sub.max(1); mn = sub.min(1)
    sat = np.where(mx>0, (mx-mn)/np.maximum(mx,1e-6), 0)
    val = mx
    r,g,b = sub[:,0],sub[:,1],sub[:,2]
    d = mx-mn
    hue = np.zeros_like(mx)
    m = d>1e-6
    idx = (mx==r)&m
    hue[idx] = (60*((g[idx]-b[idx])/d[idx]) + 360) % 360
    idx = (mx==g)&m
    hue[idx] = 60*((b[idx]-r[idx])/d[idx]) + 120
    idx = (mx==b)&m
    hue[idx] = 60*((r[idx]-g[idx])/d[idx]) + 240
    sat = np.where(m, sat, 0)
    print(f"{name:26s} n={len(sub):7d} val={val.mean():.2f} sat={sat.mean():.3f} hue={hue.mean():6.1f} "
          f"| medV={np.median(val):.2f} medS={np.median(sat):.3f} medH={np.median(hue):5.1f} "
          f"| share sat<0.12={np.mean(sat<0.12):.2f}")

print("FINAL")
hsv_stats("residual band (455-540,610-700)", 610,700,455,540)
hsv_stats("residual upper (440-500,590-640)", 590,640,440,500)
hsv_stats("blouse collar light (560-660,600-660)", 600,660,560,660)
hsv_stats("blouse body right (1000-1100,700-800)", 1000,1100,700,800)
hsv_stats("blouse ribbon (900-1000,620-720)", 900,1000,620,720)
hsv_stats("neck skin (545-580,430-520)", 545,580,430,520)
hsv_stats("hair (700-800,330-420)", 700,800,330,420)
print("ORIGINAL")
hsv_stats("orig knit under chin (610-700,455-540)", 610,700,455,540, orig)
hsv_stats("orig knit chest (950-1050,400-600)", 950,1050,400,600, orig)
hsv_stats("orig neck skin (545-580,430-520)", 545,580,430,520, orig)
