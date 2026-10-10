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

orig_p=REPO/"data/inputs/IMG_3651.jpg"; final_p=REPO/"evidence/p29-submit-run/final.png"
labels_p=REPO/"evidence/p29-collar-repair/labels_fresh.png"
original=np.asarray(Image.open(orig_p).convert("RGB"))
final=np.asarray(Image.open(final_p).convert("RGB"))
labels=np.asarray(Image.open(labels_p))
svc=NativeMaskInpainting(settings)
lm=svc._detect_landmarks(str(orig_p),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; collar=np.asarray(ms.collar)>0
background=labels==0
fw=face_width(lm); chin_x,chin_y=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(original,lm,fw); cfg=config_from_settings(settings)
skin_orig=adaptive_skin_mask(original,ref,fw,cfg)>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(original); fL,fC=LC(final)
d=np.linalg.norm(original.astype(np.float32)-final.astype(np.float32),axis=2)
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
win=np.zeros(oL.shape,bool); win[560:745,415:625]=True
resid=(fL>145)&(fC<72)&(d<12)&win&~rel&~gp
print("uncovered pale-unchanged in window:",int(resid.sum()))
cats={"protected":protected,"hair":hair,"background":background,"skin_orig":skin_orig,
      "not-in-window":~win,"not-pale-final":~((fL>145)&(fC<72))}
tot=np.zeros_like(resid)
for k,m in cats.items():
    n=int((resid&m).sum()); tot|= (resid&m); print(f"   {k:16s} {n}")
print("   none of the above (=> fails tone/skin gate only):",int((resid&~tot).sum()))
# where is the 330 px strip?
u=np.zeros(oL.shape,np.uint8); u[resid]=255
n,ll,st,_=cv2.connectedComponentsWithStats(u,8)
print("residual blobs (px, bbox) and their categories:")
for i in range(1,n):
    if st[i,4]<20: continue
    m=ll==i
    ys,xs=np.where(m)
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    hits=[k for k,mm in cats.items() if (m&mm).sum()>0.5*st[i,4]]
    print(f"   {int(st[i,4]):4d}px bbox[{x0},{y0},{x1},{y1}] protected={int((m&protected).sum())} skin={int((m&skin_orig).sum())} "
          f"oL={np.median(oL[m]):.0f} oC={np.median(oC[m]):.0f} fL={np.median(fL[m]):.0f} fC={np.median(fC[m]):.0f} d={np.median(d[m]):.1f} cats={hits}")
print("chin:",chin_x,chin_y,"face_w:",fw)
