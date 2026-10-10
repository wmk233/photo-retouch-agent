"""Show what the v2 rule leaves out of the visible cream fichu."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
OUT = ROOT/"evidence/p29-collar-repair"
orig = np.asarray(Image.open(ROOT/"data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(ROOT/"evidence/p29-submit-run/final.png").convert("RGB"))
labels = np.asarray(Image.open(OUT/"labels_fresh.png")); H,W=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(ROOT/"data/inputs/IMG_3651.jpg",orig)
ms=build_semantic_edit_masks(orig,lm,labels)
prot=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; bg=labels==0
face_w=face_width(lm); chin=lm[CHIN]; cy_,cx_=float(chin[1]),float(chin[0])
reference,_=skin_reference_colour(orig,lm,face_w); lc=config_from_settings(settings)
skin_orig=adaptive_skin_mask(orig,reference,face_w,lc)>0
a=orig.astype(np.float32); ol=0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]; oc=a.max(2)-a.min(2)
b=final.astype(np.float32); fl=0.299*b[:,:,0]+0.587*b[:,:,1]+0.114*b[:,:,2]
orig_knit=(ol>140)&(oc<50); final_bright=fl>135
win=np.zeros((H,W),bool); win[int(cy_):int(cy_+0.62*face_w), int(cx_-0.62*face_w):int(cx_+0.75*face_w)]=True
sel = orig_knit&final_bright&~skin_orig&win&~hair&~bg&~prot
missed_knit = orig_knit&win&~hair&~bg&~sel          # original knit not selected
notknit = win&~orig_knit&~hair&~bg&~prot            # bright final over non-knit original
vis = final.copy()
vis[missed_knit] = (vis[missed_knit]*0.3+np.array([0,90,255])*0.7).astype(np.uint8)
vis[sel] = (vis[sel]*0.2+np.array([255,0,0])*0.8).astype(np.uint8)
vis[notknit] = (vis[notknit]*0.6+np.array([0,255,0])*0.4).astype(np.uint8)
im = Image.fromarray(vis).crop((300,540,700,850)).resize((800,620),Image.NEAREST)
d=ImageDraw.Draw(im)
for gx in range(300,701,50):
    d.line([((gx-300)*2,0),((gx-300)*2,620)],fill=(255,255,0)); d.text(((gx-300)*2+2,2),str(gx),fill=(255,255,0))
for gy in range(550,851,50):
    d.line([(0,(gy-540)*2),(800,(gy-540)*2)],fill=(255,255,0)); d.text((2,(gy-540)*2+2),str(gy),fill=(255,255,0))
im.save(OUT/"trial_v2_diag_zoom2x.png")
# numbers in the tight fichu box
box=np.zeros((H,W),bool); box[575:715,450:600]=True
print("tight box px", int(box.sum()))
print(" sel", int((sel&box).sum()), " missed_knit", int((missed_knit&box).sum()), " notknit", int((notknit&box).sum()))
print(" missed_knit breakdown: not_bright", int(((missed_knit&box)&(fl<=135)).sum()),
      "skin_orig", int(((missed_knit&box)&skin_orig).sum()),
      "olive", int(((missed_knit&box)&~ (fl>135)).sum()))
print("ok")
