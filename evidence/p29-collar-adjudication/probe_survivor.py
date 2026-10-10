from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, fill_polygon, FACE_OVAL, skin_reference_colour)
from app.services.native_mask_inpainting import NativeMaskInpainting
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),o)
ms=build_semantic_edit_masks(o,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
oval=fill_polygon((h,w), np.asarray(lm)[list(FACE_OVAL)])>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(o); fL,fC=LC(f)
d=np.abs(o.astype(np.int16)-f.astype(np.int16)).max(2)   # per-pixel max channel change
# zone: the whole collar neighbourhood (below chin-6px, down 160px, chin +- 1.0 fw)
zone=np.zeros((h,w),bool)
zone[max(int(cy-6),0):int(cy+160), max(int(cx-1.0*fw),0):min(int(cx+1.0*fw),w)]=True
# OLD PALE KNIT in the original (clearly cloth): bright + low chroma
oldknit=(oL>150)&(oC<45)
# SURVIVOR = old pale knit, essentially unchanged by the render, in zone, not background/hair
survivor=oldknit&(d<10)&zone&~background&~hair
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
rel0=oldknit&(d<10)&zone&~background&~hair&~protected
root=cv2.dilate(rel0.astype(np.uint8),np.ones((3,3),np.uint8))>0
cand=survivor&protected&~oval
n,ll,st,_=cv2.connectedComponentsWithStats((cand*255).astype(np.uint8),8)
carve=np.zeros((h,w),bool)
for i in range(1,n):
    if ((ll==i)&root).sum()>0: carve|=ll==i
print("zone px",int(zone.sum()))
print("oldknit(orig L>150,C<45) in zone:",int((oldknit&zone).sum()))
print("SURVIVOR (old knit, unchanged) in zone:",int(survivor.sum()))
print("  survivor & protected:",int((survivor&protected).sum())," & oval:",int((survivor&oval).sum()),
      " & skin_flag:",int((survivor&(adaptive_skin_mask(o,skin_reference_colour(o,lm,fw)[0],fw,config_from_settings(settings))>0)).sum()))
print("  survivor covered by existing release:",int((survivor&rel).sum()))
print("  survivor NOT covered:",int((survivor&~rel).sum()))
print("  carve candidate (survivor&protected&~oval & connected to released survive):",int(carve.sum()))
if survivor.any():
    unc=survivor&~rel&~carve
    print("  survivor NOT covered by release|carve:",int(unc.sum()))
    n2,ll2,st2,_=cv2.connectedComponentsWithStats((unc*255).astype(np.uint8),8)
    rows=sorted([(int(st2[i,4]),i) for i in range(1,n2)],reverse=True)
    for a,i in rows[:12]:
        if a<10: break
        m=ll2==i
        print("   unc",a,"px bbox",[int(st2[i,0]),int(st2[i,1]),int(st2[i,2]),int(st2[i,3])],
              "| protected",int((m&protected).sum()),"| oval",int((m&oval).sum()),
              "| oCmed",round(float(np.median(oC[m]))),"| fCmed",round(float(np.median(fC[m]))),"| d_mean",round(float(d[m].mean()),1))
if carve.any():
    ys,xs=np.where(carve); print("  carve bbox",[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
        "oCmed",round(float(np.median(oC[carve]))),"fCmed",round(float(np.median(fC[carve]))))
