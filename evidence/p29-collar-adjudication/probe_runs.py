from __future__ import annotations
import sys, numpy as np
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
ref,_=skin_reference_colour(o,lm,fw); skin=adaptive_skin_mask(o,ref,fw,config_from_settings(settings))>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
def HS(a):
    a=a.astype(np.float32)/255.; mx,mn=a.max(2),a.min(2); dd=mx-mn
    sat=np.where(mx>1e-6,dd/np.maximum(mx,1e-6),0.)
    r,g,b=a[...,0],a[...,1],a[...,2]; hue=np.zeros_like(mx); m=dd>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/dd[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/dd[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/dd[i])+240
    return hue,np.where(m,sat,0.)
oL,oC=LC(o); fL,fC=LC(f); fH,fS=HS(f)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72)
guarded=pale&~background&~hair&~olive&skin&win
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
print("skin ref rgb",ref)
for yy in (600,615,630,645,660,675,690,705,720):
    row=guarded[yy]
    xs=np.where(row)[0]
    if len(xs)==0: print(yy,"none"); continue
    # runs
    runs=[]; s=xs[0]; p=xs[0]
    for x in xs[1:]:
        if x!=p+1: runs.append((s,p)); s=x
        p=x
    runs.append((s,p))
    runs=[r for r in runs if r[1]-r[0]>=2]
    print(f"y={yy} guarded runs {runs}")
    for (a,b) in runs[:3]:
        mid=(a+b)//2
        print(f"    run[{a},{b}] Omid{o[yy,mid].tolist()} oC{oC[yy,mid]:.0f} | Fmid{f[yy,mid].tolist()} fC{fC[yy,mid]:.0f} | left O{o[yy,a].tolist()} right O{o[yy,b].tolist()}")
