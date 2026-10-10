import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
H,W=orig.shape[:2]
skin=d['skin']; hair=d['hair']; background=d['background']; oc=d['oc']; ol=d['ol']

hsv=cv2.cvtColor(orig.astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
Hh,Ss,Vv=hsv[:,:,0]*2.0, hsv[:,:,1]/255.0, hsv[:,:,2]

def stat(name,m):
    if m.sum()<20: print(f"{name:32s} n={int(m.sum())} (too few)"); return
    h=Hh[m]; s=Ss[m]; c=oc[m]; l=ol[m]
    rgb=np.median(orig[m],0)
    print(f"{name:32s} n={int(m.sum()):6d} hue={np.median(h):5.1f} sat={np.median(s):.3f} chroma={np.median(c):5.1f} luma={np.median(l):6.1f} RGB={rgb.astype(int).tolist()}")

print("== hue/sat/chroma: fabric vs skin (ORIGINAL) ==")
stat("block 1067px", blob)
# known knit: the sweater body, split by luma (lit vs shadow)
body=(ol>140)&(oc<60)&(np.arange(H)[:,None]>820)&(skin==False)&(background==False)&(hair==False)
stat("sweater body lit (l>200)", body&(ol>200))
stat("sweater body mid (170-200)", body&(ol>170)&(ol<=200))
stat("sweater body shadow (140-170)", body&(ol>140)&(ol<=170))
stat("sweater body deep shadow (100-140)", (ol>100)&(ol<=140)&(np.arange(H)[:,None]>820)&(skin==False)&(background==False)&(hair==False))
# known skin
for nm,(x0,y0,x1,y1) in {"skin neck A [446,570]":(446,570,530,593),"skin neck B [479,618]":(479,618,510,649),
                          "skin chin [455,555]":(455,555,520,578),"skin cheek [430,470]":(430,470,470,520)}.items():
    m=np.zeros((H,W),bool); m[y0:y1,x0:x1]=True; stat(nm,m)
# the collar strip already adjudicated fabric
rel=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
stat("released strip (fabric)", rel)

print()
print("== nearest-neighbour class of the block, by (hue, chroma, luma) ==")
b_h=np.median(Hh[blob]); b_c=np.median(oc[blob]); b_l=np.median(ol[blob])
print(f"block medians hue={b_h:.1f} chroma={b_c:.1f} luma={b_l:.1f}")
def dist(m):
    return abs(np.median(Hh[m])-b_h)/10 + abs(np.median(oc[m])-b_c)/3 + abs(np.median(ol[m])-b_l)/10
cands={"sweater shadow knit":body&(ol>140)&(ol<=170),"sweater deep-shadow knit":(ol>100)&(ol<=140)&(np.arange(H)[:,None]>820)&(skin==False)&(background==False)&(hair==False),
       "released strip":rel}
for nm,(x0,y0,x1,y1) in {"skin neck A":(446,570,530,593)}.items():
    m=np.zeros((H,W),bool); m[y0:y1,x0:x1]=True; cands[nm]=m
for nm,m in cands.items(): print(f"  d({nm:24s})={dist(m):6.2f}")

print()
print("== is the block's hue the LIGHT's hue or SKIN's hue? (chroma-normalised direction) ==")
def dirn(m):
    v=np.median(orig[m],0); v=v-v.min(); s=v.sum()+1e-6
    return (v/s*100).round(1)
for nm,m in [("white sweater lit",body&(ol>200)),("white sweater shadow",body&(ol>100)&(ol<=170)&(np.arange(H)[:,None]>820)&(skin==False)),
             ("block",blob),("skin neck A",np.zeros((H,W),bool))]:
    if m.sum()<20: continue
    print(f"  {nm:22s} R:G:B% = {dirn(m)}")
m=np.zeros((H,W),bool); m[570:593,446:530]=True; print(f"  {'skin neck A':22s} R:G:B% = {dirn(m)}")
