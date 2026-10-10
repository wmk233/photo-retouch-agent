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
original=np.asarray(Image.open(orig_p).convert("RGB"))
final=np.asarray(Image.open(final_p).convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
svc=NativeMaskInpainting(settings)
lm=svc._detect_landmarks(str(orig_p),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0
fw=face_width(lm); chin_x,chin_y=(float(v) for v in lm[CHIN])
ref,_=skin_reference_colour(original,lm,fw); cfg=config_from_settings(settings)
skin_orig=adaptive_skin_mask(original,ref,fw,cfg)>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(original); fL,fC=LC(final)
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
block=np.zeros(oL.shape,bool); block[578:594,523:559]=True
below=np.zeros(oL.shape,bool); below[594:620,523:559]=True   # release blob 530@[526,594,35,25]
# chin skin strip: just above the block, on the face
chinskin=np.zeros(oL.shape,bool); chinskin[566:577,523:559]=True
for name,m in [("BLOCK330",block),("RELEASE_below_530",below&rel),("strip_above_566-577",chinskin)]:
    if m.sum()==0: print(name,"empty"); continue
    print(f"{name:22s} n={int(m.sum()):4d} oL={np.median(oL[m]):.0f} oC={np.median(oC[m]):.0f} "
          f"fL={np.median(fL[m]):.0f} fC={np.median(fC[m]):.0f} prot={int((m&protected).sum())} "
          f"skin={int((m&skin_orig).sum())} release={int((m&rel).sum())} guarded={int((m&gp).sum())}")
print("mean RGB block:",original[block].mean(0).round(1)," below:",original[below].mean(0).round(1),
      " chinskin:",original[chinskin].mean(0).round(1))
print("final mean RGB block:",final[block].mean(0).round(1)," below:",final[below].mean(0).round(1),
      " chinskin:",final[chinskin].mean(0).round(1))
# visual: original + final, tight crop, 16x
x0,x1,y0,y1=505,600,556,624
Z=16
def up(a): return cv2.resize(a,( (x1-x0)*Z,(y1-y0)*Z),interpolation=cv2.INTER_NEAREST)
oc=up(original[y0:y1,x0:x1]); fc=up(final[y0:y1,x0:x1])
# outlines on original copy
ov=oc.copy()
def outline(img,mask,color):
    mm=mask[y0:y1,x0:x1].astype(np.uint8)
    cnts,_=cv2.findContours(mm,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,[c*Z for c in cnts],-1,color,2)
outline(oc,protected,(0,0,255))     # protected red
outline(oc,rel,(255,0,0))           # release blue
outline(oc,block,(0,255,0))         # block330 green
cv2.line(oc,(0,int(chin_y-y0)*Z),((x1-x0)*Z,int(chin_y-y0)*Z),(0,255,255),2)
cv2.line(fc,(0,int(chin_y-y0)*Z),((x1-x0)*Z,int(chin_y-y0)*Z),(0,255,255),2)
sep=np.full(((y1-y0)*Z,8,3),255,np.uint8)
out=np.hstack([oc,sep,fc])
Image.fromarray(out).save(REPO/"evidence/p29-collar-adjudication/v6_block16x_outline.png")
print("saved v6_block16x_outline.png  crop",x0,x1,y0,y1,"chin_y",round(chin_y,1))
