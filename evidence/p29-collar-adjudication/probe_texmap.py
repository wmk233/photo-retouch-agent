from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour)
from app.services.native_mask_inpainting import NativeMaskInpainting
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
lab=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=lab.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),o)
ms=build_semantic_edit_masks(o,lm,lab)
hair=np.asarray(ms.hair)>0; background=lab==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(o,lm,fw)
skin=adaptive_skin_mask(o,ref,fw,config_from_settings(settings))>0
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
oL,oC=LC(o); fL,fC=LC(f); fH,fS=HS(f)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72)
guarded=pale&~background&~hair&~olive&skin&win
# original texture: local luma roughness (Laplacian energy) + 5x5 std
g=oL.astype(np.float32)
lap=np.abs(cv2.Laplacian(g,cv2.CV_32F,ksize=1))
mean=cv2.boxFilter(g,-1,(5,5))
sq=cv2.boxFilter(g*g,-1,(5,5))
std5=np.sqrt(np.maximum(sq-mean*mean,0))
# references
sweater=win&~skin&~background&~hair&~olive&(oL>150)&(oC<45)   # released knit
neck=  np.zeros((h,w),bool); neck[610:700,455:495]=True           # neck skin band
print("sweater px",int(sweater.sum()),"neck ref px",int(neck.sum()),"guarded",int(guarded.sum()))
for nm,m in [("sweater(released knit)",sweater),("neck ref",neck),("guarded",guarded),("guarded|release",guarded|(win&~skin&~background&~hair&~olive))]:
    if m.sum()==0: continue
    print(f"{nm:24s} lap p10/50/90={np.percentile(lap[m],[10,50,90]).round(2)} std5 p10/50/90={np.percentile(std5[m],[10,50,90]).round(2)} oLmed={np.median(oL[m]):.0f} oCmed={np.median(oC[m]):.0f}")
# spatial split of guarded by texture
gs=std5[guarded]
print("guarded std5 quartiles", np.percentile(gs,[25,50,75]).round(2))
# Save a heat map of std5 over the crop with guarded outline
x0,x1,y0,y1=430,600,560,715
Z=8
crop=lambda a:a[y0:y1,x0:x1]
rgb=np.zeros((y1-y0,x1-x0,3),np.uint8)
t=std5[y0:y1,x0:x1]
tn=np.clip(t/8.0,0,1)
rgb[...,0]=(tn*255).astype(np.uint8)
rgb[...,1]=(np.clip((t-2)/6.0,0,1)*255).astype(np.uint8)
base=crop(f).copy()
base[guarded[y0:y1,x0:x1]]=(0.4*base[guarded[y0:y1,x0:x1]]+0.6*np.array([0,0,255])).astype(np.uint8)
row=np.concatenate([crop(o),base,rgb],axis=1)
big=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(REPO/"evidence/p29-collar-adjudication/texmap_8x.png")
print("saved texmap_8x.png")
