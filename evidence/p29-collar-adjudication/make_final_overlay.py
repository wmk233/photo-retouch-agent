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
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
band=np.zeros((h,w),bool)
band[max(int(round(cy-0.10*fw)),0):int(round(cy)), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
region=win|band
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72); sknit=(oL>140)&(oC<50)
resid=pale&sknit&~skin&~olive&~hair&~background&region
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
rel0=resid&~protected
cand=resid&protected
root=cv2.dilate(rel0.astype(np.uint8),np.ones((3,3),np.uint8))>0
n2,ll2,st2,_=cv2.connectedComponentsWithStats((cand*255).astype(np.uint8),8)
carve=np.zeros((h,w),bool)
for i in range(1,n2):
    m=ll2==i
    if (m&root).sum()>0: carve|=m
guarded=pale&~background&~hair&~olive&skin&region
uncovered=resid&~(rel|carve)
x0,x1,y0,y1=400,690,555,765
Z=5
def crop(a): return a[y0:y1,x0:x1]
base=crop(final).copy()
ov=base.copy()
def edge(m):
    u=m.astype(np.uint8)
    er=cv2.erode(u,np.ones((3,3),np.uint8))
    return (u>0)&(er==0)
for m,col in ((crop(protected),(255,255,0)),(crop(rel),(255,60,60)),
              (crop(carve),(60,255,60)),(crop(guarded),(255,0,255)),
              (crop(uncovered),(0,128,255))):
    ov[edge(m)]=col
cv2.line(ov,(0,int(cy)-y0),(x1-x0,int(cy)-y0),(0,255,255),1)
out=np.concatenate([base,ov],axis=1)
out=cv2.resize(out,(out.shape[1]*Z,out.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(out).save(REPO/"evidence/p29-collar-adjudication/final_release_carve_uncovered_5x.png")
print("saved", out.shape, "crop x",x0,x1,"y",y0,y1)
print("resid",int(resid.sum()),"carve",int(carve.sum()),"uncovered",int(uncovered.sum()),"guarded",int(guarded.sum()))
print("legend: red=existing release, green=carve, magenta=guarded_pale, orange=uncovered resid, yellow=protected boundary, cyan=chin_y")
