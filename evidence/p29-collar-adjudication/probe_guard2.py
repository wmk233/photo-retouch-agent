from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
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
oL,oC=LC(original); fL,fC=LC(final)
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
print("guarded n:",int(gp.sum()))
for q in (5,25,50,75,95):
    print(f"   p{q:2d}  oL={np.percentile(oL[gp],q):6.0f} oC={np.percentile(oC[gp],q):5.0f} fL={np.percentile(fL[gp],q):6.0f} fC={np.percentile(fC[gp],q):5.0f}")
print("   in protected:",int((gp&protected).sum())," in background:",int((gp&background).sum())," skinflag:",int((gp&skin).sum()))
# how white is it in the FINAL? bucket by final luma
for lo,hi in ((132,150),(150,170),(170,185),(185,200),(200,256)):
    m=gp&(fL>=lo)&(fL<hi); print(f"   final L {lo}-{hi}: {int(m.sum()):5d} px  median fC={np.median(fC[m]) if m.sum() else 0:.0f}")
# distinct blobs
n,ll,st,_=cv2.connectedComponentsWithStats((gp*255).astype(np.uint8),8)
bl=sorted([(int(st[i,4]),[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]) for i in range(1,n)],reverse=True)
print("guarded blobs:",bl[:10])
# VISUAL: final crop of the collar, 6x, with guarded outline and release outline
x0,x1,y0,y1=400,620,560,760
Z=6
def crop(a): return a[y0:y1,x0:x1]
fin=crop(final).copy(); ori=crop(original).copy()
def outline(img,mask,color):
    mm=mask[y0:y1,x0:x1].astype(np.uint8)
    cnts,_=cv2.findContours(mm,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,[c for c in cnts],-1,color,1)
for img in (fin,ori):
    pass
outline(fin,rel,(255,0,0)); outline(fin,gp,(0,0,255))
outline(ori,rel,(255,0,0)); outline(ori,gp,(0,0,255))
for img in (fin,ori):
    cv2.line(img,(0,int(cy)-y0),((x1-x0),int(cy)-y0),(0,255,255),1)
fin=cv2.resize(fin,((x1-x0)*Z,(y1-y0)*Z),interpolation=cv2.INTER_NEAREST)
ori=cv2.resize(ori,((x1-x0)*Z,(y1-y0)*Z),interpolation=cv2.INTER_NEAREST)
sep=np.full(((y1-y0)*Z,10,3),255,np.uint8)
Image.fromarray(np.hstack([ori,sep,fin])).save(REPO/"evidence/p29-collar-adjudication/v7_orig_final_release_guarded_6x.png")
print("saved v7_orig_final_release_guarded_6x.png crop",x0,x1,y0,y1)
