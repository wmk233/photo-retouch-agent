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
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(original,lm,fw); skin=adaptive_skin_mask(original,ref,fw,config_from_settings(settings))>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(original); fL,fC=LC(final)
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
print("row profile x523:559, y568..604  (median oL / oC / n_protected / n_skin)")
for y in range(568,605):
    row=slice(y,y+1),slice(523,559)
    print(f"  y={y:3d} oL={np.median(oL[row]):5.0f} oC={np.median(oC[row]):4.0f} "
          f"prot={int(protected[row].sum()):2d} skin={int(skin[row].sum()):2d} rel={int(rel[row].sum()):2d} gp={int(gp[row].sum()):2d}")
# the 330 block exactly
block=np.zeros(oL.shape,bool); block[578:594,523:559]=True
m=block&protected
print("in-box protected px:",int(m.sum()))
for tag,mm in [("all_box_prot",m),("box_prot_skin",m&skin),("box_prot_NONskin",m&~skin)]:
    if mm.sum()==0: print(f"  {tag}: 0"); continue
    print(f"  {tag:16s} n={int(mm.sum()):4d} oL med={np.median(oL[mm]):.0f} oC med={np.median(oC[mm]):.0f} "
          f"oL p10={np.percentile(oL[mm],10):.0f} oC p90={np.percentile(oC[mm],90):.0f} fL med={np.median(fL[mm]):.0f}")
