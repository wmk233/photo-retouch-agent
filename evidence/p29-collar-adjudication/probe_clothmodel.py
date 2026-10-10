from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour, FACE_OVAL, fill_polygon)
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
oval=fill_polygon((h,w),np.asarray(lm)[list(FACE_OVAL)])>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(o); fL,fC=LC(f)
d=np.abs(o.astype(np.int16)-f.astype(np.int16)).max(2)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
# cloth training: released knit, unchanged, bright, unmixed
cloth=rel&(d<10)&(oL>150)
skintr=oval&skin&(d<10)&~hair
print("cloth train",int(cloth.sum()),"skin train",int(skintr.sum()))
# feature: normalized rgb (r,g,b)/(sum)
def feat(a):
    x=a.astype(np.float32)+1.0
    s=x.sum(2,keepdims=True)
    return x/s
F=feat(o)
C=F[cloth]; S=F[skintr]
muC=C.mean(0); covC=np.cov(C.T)+np.eye(3)*1e-5
muS=S.mean(0); covS=np.cov(S.T)+np.eye(3)*1e-5
iC=np.linalg.inv(covC); iS=np.linalg.inv(covS)
def maha(x,mu,ic):
    dx=x-mu
    return np.sqrt(np.einsum('...i,ij,...j->...',dx,ic,dx))
dC=maha(F,muC,iC); dS=maha(F,muS,iS)
print("cloth mu",muC.round(4),"skin mu",muS.round(4))
print("on cloth train: dC med",np.median(dC[cloth]).round(2),"dS med",np.median(dS[cloth]).round(2))
print("on skin  train: dC med",np.median(dC[skintr]).round(2),"dS med",np.median(dS[skintr]).round(2))
guarded_np=np.load(REPO/"evidence/p29-collar-adjudication/_ot.npy") if False else None
pale=(fL>132)&(fC>25)&(fC<72)
def HS(a):
    a=a.astype(np.float32)/255.; mx,mn=a.max(2),a.min(2); dd=mx-mn
    sat=np.where(mx>1e-6,dd/np.maximum(mx,1e-6),0.)
    r,g,b=a[...,0],a[...,1],a[...,2]; hue=np.zeros_like(mx); m=dd>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/dd[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/dd[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/dd[i])+240
    return hue,np.where(m,sat,0.)
fH,fS=HS(f); olive=(fH>=45)&(fH<=115)&(fS>0.12)
guarded=pale&~background&~hair&~olive&skin&win
print("guarded",int(guarded.sum()))
gC=dC[guarded]; gS=dS[guarded]
print("guarded dC p10/50/90",np.percentile(gC,[10,50,90]).round(2))
print("guarded dS p10/50/90",np.percentile(gS,[10,50,90]).round(2))
closer_cloth=(dC<dS)
print("guarded closer-to-cloth:",int((guarded&closer_cloth).sum()),"/",int(guarded.sum()))
# how confident: require cloth to be clearly closer
for margin in (0.0,0.5,1.0,2.0):
    m=guarded&(dS-dC>margin)
    print(f"  margin>{margin}: {int(m.sum())} px, &oval {int((m&oval).sum())}, &skinref-diff ... ")
print("closer_cloth within oval (sanity):",int((oval&closer_cloth).sum()),"/",int(oval.sum()))
print("closer_cloth within hair:",int((hair&closer_cloth).sum()),"/",int(hair.sum()))
# where is closer_cloth in guarded?
m=guarded&closer_cloth
ys,xs=np.where(m)
if m.any(): print("bbox",[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
