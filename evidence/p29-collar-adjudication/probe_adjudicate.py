from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
def LC(a):
    a=a.astype(np.float32)
    return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
orig=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
lab=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
noedit=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_no_edit.png"))>127
oL,oC=LC(orig); fL,fC=LC(final)
d=np.linalg.norm(orig.astype(np.float32)-final.astype(np.float32),axis=2)

W=np.zeros(orig.shape[:2],bool); W[590:735,505:600]=True

print("--- SCHP labels inside gp ---")
vals,cts=np.unique(lab[gp],return_counts=True)
print(dict(zip(vals.tolist(),cts.tolist())))
print("--- SCHP labels in box x520-580 y600-700 ---")
b=lab[600:700,520:580]; v,c=np.unique(b,return_counts=True); print(dict(zip(v.tolist(),c.tolist())))

print("--- inside gp ---")
print("orig L range", oL[gp].min().round(0), oL[gp].max().round(0), "C range", oC[gp].min().round(0), oC[gp].max().round(0))
for t in (55,60,65,70):
    print(f"  gp px with orig C >= {t}: {int((gp&(oC>=t)).sum())}")
print("  gp px with |final-orig| > 12:", int((gp&(d>12)).sum()))
print("  gp px with orig L < 145:", int((gp&(oL<145)).sum()))
print("  gp px touching noedit (dilate1):", int((gp&cv2.dilate(noedit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)).sum()))
print("  gp px within 2px of noedit:", int((gp&(cv2.dilate(noedit.astype(np.uint8),np.ones((5,5),np.uint8))>0)).sum()))

orig_knit=(oL>145)&(oC<52)
print("--- white-knit mass (orig L>145 & C<52) in collar window y560-740 x420-620 ---")
K=np.zeros(orig.shape[:2],bool); K[560:740,420:620]=True
km=orig_knit&K
print("  total white-knit px in window:", int(km.sum()))
print("  of which in release:", int((km&rel).sum()))
print("  of which in guarded:", int((km&gp).sum()))
print("  of which in neither:", int((km&~rel&~gp).sum()))
ys,xs=np.where(km&~rel&~gp)
if len(ys): print("  neither bbox:", xs.min(),ys.min(),xs.max(),ys.max())
# --- residual coverage in the FINAL: pale cloth still standing, unchanged ---
win=np.zeros(orig.shape[:2],bool); win[560:745,415:625]=True
unchanged=d<12
resid=(fL>145)&(fC<72)&unchanged&win
print("--- FINAL pale-unchanged residual in collar window ---")
print("  total:", int(resid.sum()), " in release:", int((resid&rel).sum()), " in guarded:", int((resid&gp).sum()))
print("  neither:", int((resid&~rel&~gp).sum()))
ys,xs=np.where(resid&~rel&~gp)
if len(ys):
    print("  neither bbox:", xs.min(),ys.min(),xs.max(),ys.max())
    # blob sizes
    u8=np.zeros_like(lab,np.uint8); u8[resid&~rel&~gp]=255
    n,ll,st,_=cv2.connectedComponentsWithStats(u8,8)
    blobs=sorted([(int(st[i,cv2.CC_STAT_AREA]),[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]) for i in range(1,n)],reverse=True)
    print("  blobs:", blobs[:8])
# --- connectivity of pale mass ---
pale=(fL>145)&(fC<72)
u8=(pale*255).astype(np.uint8); u8=cv2.morphologyEx(u8,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,labels,stats,_=cv2.connectedComponentsWithStats(u8,8)
best=-1;bestn=0
for i in range(1,n):
    m=labels==i; g=int((m&gp).sum()); r=int((m&rel).sum())
    if g>100 and r>100: best=i;bestn=g
    print(f"  comp {i}: area {int(stats[i,cv2.CC_STAT_AREA])} gp {g} rel {r} bbox {[int(v) for v in stats[i,:4]]}")
print("guarded px in the joint component:", bestn, "of", int(gp.sum()))
