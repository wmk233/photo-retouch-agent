import numpy as np
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
skin=d['skin']; ol=d['ol']; oc=d['oc']; fl=d['fl']; fc=d['fc']
for y in (600, 620, 640, 660, 680):
    print(f"--- scanline y={y} : x 470..600 ---")
    print(" x   origRGB          ol  oc  | finRGB           fl  fc  | skin blob")
    for x in range(470, 601, 4):
        o=orig[y,x]; f=fin[y,x]
        print(f"{x:4d} {str(o.tolist()):16s} {ol[y,x]:5.0f} {oc[y,x]:4.0f} | {str(f.tolist()):16s} {fl[y,x]:5.0f} {fc[y,x]:4.0f} | {'S' if skin[y,x] else '.'}    {'B' if blob[y,x] else '.'}")
