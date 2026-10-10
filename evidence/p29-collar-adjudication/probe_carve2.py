from __future__ import annotations
import sys, numpy as np, cv2
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

PALE_LUMA_MIN,PALE_C_MIN,PALE_C_MAX=132.,25.,72.
SOURCE_L_MIN,SOURCE_C_MAX=140.,50.
OL_H0,OL_H1,OL_S=45.,115.,0.12
WD,WL,WR=0.62,0.62,0.75
MIN_BLOB_PX=400
pale=(fL>PALE_LUMA_MIN)&(fC>PALE_C_MIN)&(fC<PALE_C_MAX)
sknit=(oL>SOURCE_L_MIN)&(oC<SOURCE_C_MAX)
olive=(fH>=OL_H0)&(fH<=OL_H1)&(fS>OL_S)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+WD*fw), max(int(cx-WL*fw),0):min(int(cx+WR*fw),w)]=True
CARVE_UP=0.10
band=np.zeros((h,w),bool)
band[max(int(round(cy-CARVE_UP*fw)),0):int(round(cy)), max(int(cx-WL*fw),0):min(int(cx+WR*fw),w)]=True
print("carve band px:",int(band.sum()),"y",int(round(cy-CARVE_UP*fw)),"..",int(round(cy)),
      "x",max(int(cx-WL*fw),0),"..",min(int(cx+WR*fw),w))
resid_ext=pale&sknit&~skin&(win|band)&~hair&~background&~olive
rel0=(resid_ext&~protected)
cand=(resid_ext&protected)
print("resid_ext",int(resid_ext.sum()),"rel0",int(rel0.sum()),"cand",int(cand.sum()))
root=cv2.dilate(rel0.astype(np.uint8),np.ones((3,3),np.uint8))>0
u=(cand*255).astype(np.uint8)
n,ll,st,_=cv2.connectedComponentsWithStats(u,8)
carve=np.zeros((h,w),bool)
for i in range(1,n):
    m=ll==i
    if (m&root).sum()>0: carve|=m
print("carve (connected to release):",int(carve.sum()))
print("  carve & skin_flag:",int((carve&skin).sum())," carve & ~oval:",int((carve&~oval).sum()),
      " carve & oval:",int((carve&oval).sum()))
if carve.any():
    print("  carve tone: oL med",round(float(np.median(oL[carve]))),"oC med",round(float(np.median(oC[carve]))),
          " fL",round(float(np.median(fL[carve]))),"fC",round(float(np.median(fC[carve]))))
    ys,xs=np.where(carve); print("  carve bbox",[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
release=(rel0|carve)
release=cv2.dilate(release.astype(np.uint8),np.ones((3,3),np.uint8))>0
release=release&~protected&~hair&~background
# after dilation, re-add carve (dilation is masked by ~protected)
release=(rel0|carve)
print("FINAL release:",int(release.sum()))
# coverage: visible pale residual left
gp_pale=(pale&win&~background&~hair&~olive&skin)
left=pale&(win|band)&~background&~hair&~release
print("pale in window+band not released:",int(left.sum()))
u2=(left*255).astype(np.uint8)
n2,ll2,st2,_=cv2.connectedComponentsWithStats(u2,8)
blobs=sorted([(int(st2[i,4]),[int(st2[i,0]),int(st2[i,1]),int(st2[i,2]),int(st2[i,3])]) for i in range(1,n2)],reverse=True)
for a,b in blobs[:8]:
    if a<10: break
    m=ll2==0
    print("   leftover",a,"px bbox",b)
