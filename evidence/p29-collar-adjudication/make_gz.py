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
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
hair=np.asarray(ms.hair)>0; background=labels==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(original,lm,fw); skin=adaptive_skin_mask(original,ref,fw,config_from_settings(settings))>0
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
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72)
guarded=pale&~background&~hair&~olive&skin&win
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
whitish=(fL>168)&(fC<45)&win&~background&~hair&~olive
x0,x1,y0,y1=430,600,585,700
Z=8
def crop(a): return a[y0:y1,x0:x1].copy()
ov=crop(final)
ov[guarded[y0:y1,x0:x1]]=(70,70,255)
ov[rel[y0:y1,x0:x1]]=(255,50,50)
ov[whitish[y0:y1,x0:x1]]=(0,220,0)
row=np.concatenate([crop(original),crop(final),ov],axis=1)
big=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(REPO/"evidence/p29-collar-adjudication/gz_labeled_8x.png")
print("saved gz_labeled_8x.png crop",x0,x1,y0,y1,"Z",Z)
# row scans
for yy in (600,620,640,660,680):
    print(f"--- y={yy} ---")
    for xx in range(440,575,10):
        print(f"  x{xx} O{original[yy,xx].tolist()} F{final[yy,xx].tolist()} g{int(guarded[yy,xx])} r{int(rel[yy,xx])} wht{int(whitish[yy,xx])}")
