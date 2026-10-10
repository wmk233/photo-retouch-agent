from __future__ import annotations
import sys, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (FACE_OVAL, adaptive_skin_mask, config_from_settings,
    face_width, fill_polygon, skin_reference_colour)
from app.services.native_mask_inpainting import NativeMaskInpainting
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(original,lm,fw); skin=adaptive_skin_mask(original,ref,fw,config_from_settings(settings))>0
oval=fill_polygon((h,w), np.asarray(lm)[list(FACE_OVAL)])>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
def HS(a):
    a=a.astype(np.float32)/255.; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.)
    r,g,b=a[...,0],a[...,1],a[...,2]; hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue,np.where(m,sat,0.)
oL,oC=LC(original); fL,fC=LC(final); fH,fS=HS(final)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
band=np.zeros((h,w),bool)
band[max(int(round(cy-0.10*fw)),0):int(round(cy)), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
region=win|band
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72); sknit=(oL>140)&(oC<50)
resid=pale&sknit&~skin&~olive&~hair&~background&region   # exactly the builder residual
print("resid EXT px:",int(resid.sum()))
guarded=pale&~background&~hair&~olive&skin&region
print("guarded_pale px:",int(guarded.sum()), "guarded fC med",round(float(np.median(fC[guarded]))),
      "fL med",round(float(np.median(fL[guarded]))), "oC med",round(float(np.median(oC[guarded]))))
rel0=resid&~protected
cand=resid&protected
print("rel0",int(rel0.sum()),"cand",int(cand.sum()))
root=cv2.dilate(rel0.astype(np.uint8),np.ones((3,3),np.uint8))>0
n2,ll2,st2,_=cv2.connectedComponentsWithStats((cand*255).astype(np.uint8),8)
carve=np.zeros((h,w),bool)
for i in range(1,n2):
    m=ll2==i
    if (m&root).sum()>0: carve|=m
print("carve",int(carve.sum()),"carve&oval",int((carve&oval).sum()),"carve&skin",int((carve&skin).sum()))
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
print("existing mask px:",int(rel.sum()))
print("resid NOT covered by existing release:",int((resid&~rel).sum()))
nm=resid&~(rel|carve)
print("resid NOT covered by release|carve:",int(nm.sum()))
n3,ll3,st3,_=cv2.connectedComponentsWithStats((nm*255).astype(np.uint8),8)
rows=sorted([(int(st3[i,4]),i) for i in range(1,n3)],reverse=True)
print("uncovered resid blobs:")
for a,i in rows[:15]:
    if a<10: break
    m=ll3==i
    print("   ",a,"px bbox",[int(st3[i,0]),int(st3[i,1]),int(st3[i,2]),int(st3[i,3])],
          "| protected",int((m&protected).sum()),"| skin",int((m&skin).sum()),
          "| oCmed",round(float(np.median(oC[m]))),"| fCmed",round(float(np.median(fC[m]))),
          "| fLmed",round(float(np.median(fL[m]))))
