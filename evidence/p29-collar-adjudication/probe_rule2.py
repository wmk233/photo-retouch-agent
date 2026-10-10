from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
noedit=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_no_edit.png"))>127
oL,oC=LC(o); fL,fC=LC(f)

# 1) distribution of original chroma / luma inside guarded
print("gp n =", int(gp.sum()))
for q in (5,10,25,50,75,90,95):
    print(f"  gp orig L p{q}={np.percentile(oL[gp],q):.0f}  C p{q}={np.percentile(oC[gp],q):.0f}  final L p{q}={np.percentile(fL[gp],q):.0f}  final C p{q}={np.percentile(fC[gp],q):.0f}")
# 2) pale-mass connectivity
pale=(fL>145)&(fC<72)
u8=(pale*255).astype(np.uint8); u8=cv2.morphologyEx(u8,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,ll,st,_=cv2.connectedComponentsWithStats(u8,8)
mass=np.zeros_like(pale)
for i in range(1,n):
    m=ll==i
    if (m&rel).sum()>200 and (m&gp).sum()>200: mass|=m
print("mass area",int(mass.sum()),"gp in mass",int((mass&gp).sum()),"rel in mass",int((mass&rel).sum()))

# 3) skin evidence in the original: high-chroma skin pixels near the collar
skinhi = oC>=70
print("orig skin C>=70 px total (window)", int((skinhi & np.pad(np.ones((200,300),bool),((560,1190-760),(400,904-700))) ).sum()))
# distance (in px) from each guarded pixel to the nearest orig skin pixel
dist = cv2.distanceTransform((~skinhi).astype(np.uint8), cv2.DIST_L2, 3)
dc = dist[gp]
print("gp dist-to-skin(C>=70): p5=%.1f p25=%.1f p50=%.1f p75=%.1f p95=%.1f max=%.1f"%((*[np.percentile(dc,q) for q in (5,25,50,75,95)], dc.max())))
for t in (1,2,3,4,6,8):
    print(f"  gp px with dist-to-skin > {t}: {int((gp&(dist>t)).sum())}")

# 4) how many guarded px sit inside the identity 'protected' core? (noedit minus olive/skin)
#    Approximate: guards = noedit & ~olive & ~skin_original_window
hue_f = None
# recompute olive on final
a=f.astype(np.float32)/255.0
mx,mn=a.max(2),a.min(2); d_=mx-mn
sat=np.where(mx>1e-6,d_/np.maximum(mx,1e-6),0.0)
r,g,b=a[...,0],a[...,1],a[...,2]
h=np.zeros_like(mx); m=d_>1e-6
i=(mx==r)&m; h[i]=(60*((g[i]-b[i])/d_[i])+360)%360
i=(mx==g)&m; h[i]=60*((b[i]-r[i])/d_[i])+120
i=(mx==b)&m; h[i]=60*((r[i]-g[i])/d_[i])+240
olive=(h>=45)&(h<=115)&(sat>0.12)
print("olive px inside gp:", int((gp&olive).sum()))
print("noedit px inside gp:", int((gp&noedit).sum()))
print("noedit&~olive inside gp:", int((gp&noedit&~olive).sum()))
