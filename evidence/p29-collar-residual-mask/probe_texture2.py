import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
blob=np.load(f"{WORK}/_blob.npy"); sample=np.load(f"{WORK}/_sample.npy")
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("L")).astype(np.float32)
H,W=orig.shape
# LOCAL texture energy: high-pass (pixel - 5x5 gaussian), robust energy = median |hp|
hp=orig-cv2.GaussianBlur(orig,(0,0),1.6)
def energy(m):
    v=np.abs(hp[m]); return np.median(v), np.percentile(v,90)
def box(x0,y0,w,h):
    m=np.zeros((H,W),bool); m[y0:y0+h,x0:x0+w]=True; return m
regions={
 "KNOWN white knit body  [610,700,60,60]": box(610,700,60,60),
 "KNOWN knit fold        [560,600,30,40]": box(560,600,30,40),
 "KNOWN neck skin (side) [455,615,40,40]": box(455,615,40,40),
 "KNOWN cheek skin       [430,470,40,40]": box(430,470,40,40),
 "BLOCK 1067 (all)       [514,578,51,111]": blob,
 "BLOCK released 313px": (blob&sample),
 "BLOCK protected 754px": (blob&~sample),
}
print("local high-pass |hp| median / p90  (higher = more texture/structure)")
for nm,m in regions.items():
    md,p90=energy(m); print("  %-40s n=%5d  med=%.2f p90=%.2f"%(nm,int(m.sum()),md,p90))
print()
print("same-region chroma p50 and luma p50 (from the ORIGINAL):")
col=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
oc=col.max(2)-col.min(2)
for nm,m in regions.items():
    print("  %-40s chroma p50=%5.1f   luma p50=%5.0f"%(nm,np.median(oc[m]),np.median(orig[m])))
