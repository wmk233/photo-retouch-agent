"""v2: garment evidence is taken from the ORIGINAL (knit), skin guard from the ORIGINAL."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
OUT = ROOT/"evidence/p29-collar-repair"
orig = np.asarray(Image.open(ROOT/"data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(ROOT/"evidence/p29-submit-run/final.png").convert("RGB"))
labels = np.asarray(Image.open(OUT/"labels_fresh.png")); H,W = labels.shape
svc = NativeMaskInpainting(settings); lm = svc._detect_landmarks(ROOT/"data/inputs/IMG_3651.jpg", orig)
ms = build_semantic_edit_masks(orig, lm, labels)
prot=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; bg=labels==0
collar=np.asarray(ms.collar)>0; cloth=np.asarray(ms.clothing)>0
face_w=face_width(lm); chin=lm[CHIN]; cy_,cx_=float(chin[1]),float(chin[0])
reference,_=skin_reference_colour(orig,lm,face_w); lc=config_from_settings(settings)
skin_orig=adaptive_skin_mask(orig,reference,face_w,lc)>0
skin_final=adaptive_skin_mask(final,reference,face_w,lc)>0
def lc2(img):
    a=img.astype(np.float32); return (0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2))
ol,oc=lc2(orig); fl,fc=lc2(final)
a=(final/255.0); mx=a.max(2); mn=a.min(2); d=mx-mn
sat=np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0); r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
hue=np.zeros_like(mx); m=d>1e-6
i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
olive=(hue>=40)&(hue<=115)&(sat>0.15)
win=np.zeros((H,W),bool); win[int(cy_):int(cy_+0.62*face_w), int(cx_-0.62*face_w):int(cx_+0.75*face_w)]=True
orig_knit=(ol>140)&(oc<50); final_bright=fl>135
base=orig_knit&final_bright&~skin_orig&win&~hair&~bg&~olive
repair=base&~prot; conflict=base&prot
u8=(repair*255).astype(np.uint8)
u8=cv2.morphologyEx(u8,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
u8=cv2.morphologyEx(u8,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(u8,8)
keep=np.zeros_like(u8)
for i in range(1,n):
    if st[i,cv2.CC_STAT_AREA]>=400: keep[lab==i]=255
rep=keep>0
rep= cv2.dilate(rep.astype(np.uint8), np.ones((3,3),np.uint8))>0
rep = rep & ~prot & ~hair & ~bg
print(json.dumps({
 "window_px":int(win.sum()), "orig_knit_in_window":int((orig_knit&win).sum()),
 "base_px":int(base.sum()), "repair_px":int(rep.sum()),
 "conflict_protected_px":int(conflict.sum()),
 "excluded_skin_orig":int((orig_knit&final_bright&skin_orig&win&~hair&~bg).sum()),
 "excluded_olive":int((orig_knit&final_bright&olive&win&~hair&~bg).sum()),
 "excluded_not_bright":int((orig_knit&~final_bright&win&~hair&~bg).sum()),
 "ambiguous_pale_in_final_skin":int((rep&skin_final).sum()),
 "repair_bbox":[int(np.where(rep.any(0))[0].min()),int(np.where(rep.any(1))[0].min()),
                int(np.where(rep.any(0))[0].max()),int(np.where(rep.any(1))[0].max())] if rep.any() else None,
}, indent=1))
vis=final.copy(); vis[rep]=(vis[rep]*0.2+np.array([255,0,0])*0.8).astype(np.uint8)
vis[conflict]=(vis[conflict]*0.2+np.array([255,255,0])*0.8).astype(np.uint8)
Image.fromarray(vis).crop((300,540,700,850)).resize((800,620),Image.NEAREST).save(OUT/"trial_v2_zoom2x.png")
Image.fromarray(keep).save(OUT/"trial_v2_mask.png")
print("ok")
