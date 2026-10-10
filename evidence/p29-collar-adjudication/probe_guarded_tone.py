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
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
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
band=np.zeros((h,w),bool)
band[max(int(round(cy-0.10*fw)),0):int(round(cy)), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
region=win|band
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72); sknit=(oL>140)&(oC<50)
guarded=pale&~background&~hair&~olive&skin&win
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
# known skin: chin band just above the collar, inside oval, flagged skin, in window y cy-30..cy-8
chin_skin=np.zeros((h,w),bool)
chin_skin[int(cy-34):int(cy-6), int(cx-40):int(cx+40)]=True
chin_skin&=skin&~guarded&pale
def stats(m,name):
    if not m.any(): print(name,"EMPTY"); return
    print(f"{name:22s} px={int(m.sum()):5d}  ORIG rgb={np.round(original[m].mean(0)).astype(int)}  luma={round(float(np.median(oL[m])))} chroma(med)={round(float(np.median(oC[m])))}")
    print(f"{'':22s}            FINAL rgb={np.round(final[m].mean(0)).astype(int)}  luma={round(float(np.median(fL[m])))} chroma(med)={round(float(np.median(fC[m])))}")
stats(guarded,"guarded_pale")
stats(rel&pale,"release(pale)")
stats(chin_skin,"chin skin(above)")
# hue of guarded in final
gh=fH[guarded]; print("guarded final hue p10/50/90:",[round(float(np.percentile(gh,q))) for q in (10,50,90)])
# per-row profile of guarded
ys,xs=np.where(guarded)
print("guarded bbox",[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
# 8x panels final/original/final+contours
x0,x1,y0,y1=int(cx-70),int(cx+90),int(cy-40),int(cy+150)
Z=6
def crop(a): return a[y0:y1,x0:x1]
f=crop(final).copy(); o=crop(original).copy(); ov=crop(final).copy()
def edge(m):
    u=m.astype(np.uint8); return (u>0)&(cv2.erode(u,np.ones((3,3),np.uint8))==0)
ov[edge(crop(protected))]=(255,255,0)
ov[edge(crop(rel))]=(255,60,60)
ov[edge(crop(guarded))]=(255,0,255)
cv2.line(ov,(0,int(cy)-y0),(x1-x0,int(cy)-y0),(0,255,255),1)
row=np.concatenate([o,f,ov],axis=1)
row=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(row).save(REPO/"evidence/p29-collar-adjudication/guarded_tone_6x_orig_final_ov.png")
print("saved crop x",x0,x1,"y",y0,y1)
