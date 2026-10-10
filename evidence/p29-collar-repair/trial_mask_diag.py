"""Why is only a sliver selected? Break the residual patch down by rule."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
OUT = ROOT / "evidence/p29-collar-repair"
orig = np.asarray(Image.open(ROOT/"data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(ROOT/"evidence/p29-submit-run/final.png").convert("RGB"))
labels = np.asarray(Image.open(OUT/"labels_fresh.png"))
H, W = labels.shape
svc = NativeMaskInpainting(settings); lm = svc._detect_landmarks(ROOT/"data/inputs/IMG_3651.jpg", orig)
ms = build_semantic_edit_masks(orig, lm, labels)
prot = np.asarray(ms.protected) > 0; hair = np.asarray(ms.hair) > 0; cloth = np.asarray(ms.clothing) > 0
collar = np.asarray(ms.collar) > 0; parsed_face = np.asarray(ms.parsed_face) > 0
bg = labels == 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(orig, lm, face_w)
skin = adaptive_skin_mask(final, reference, face_w, config_from_settings(settings)) > 0
a = final.astype(np.float32); fl = 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]; fc = a.max(2)-a.min(2)
b = orig.astype(np.float32); ol = 0.299*b[:,:,0]+0.587*b[:,:,1]+0.114*b[:,:,2]; oc = b.max(2)-b.min(2)
mx=(final/255.0).max(2); mn=(final/255.0).min(2); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0)
r,g,bb=(final/255.0)[:,:,0],(final/255.0)[:,:,1],(final/255.0)[:,:,2]; d=mx-mn; hue=np.zeros_like(mx); m=d>1e-6
i=(mx==r)&m; hue[i]=(60*((g[i]-bb[i])/d[i])+360)%360
i=(mx==g)&m; hue[i]=60*((bb[i]-r[i])/d[i])+120
i=(mx==bb)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
olive=(hue>=40)&(hue<=115)&(sat>0.15)

box = np.zeros((H,W), bool); box[585:715, 455:585] = True
tot = box.sum()
print(f"box px={tot}")
for name, m in [("orig_knit(ol>140,oc<50)",(ol>140)&(oc<50)),("final_bright(>135)",fl>135),
                ("skin_final",skin),("olive_final",olive),("hair",hair),("bg",bg),("prot",prot),
                ("cloth",cloth),("collar",collar),("parsed_face",parsed_face)]:
    print(f"  {name:26s} {int((m&box).sum()):7d} ({int((m&box).sum())/tot:.0%})")
print("inter: orig_knit & final_bright:", int(((ol>140)&(oc<50)&(fl>135)&box).sum()))
inter = (ol>140)&(oc<50)&(fl>135)&box
print("  of that: skin", int((inter&skin).sum()), "olive", int((inter&olive).sum()), "prot", int((inter&prot).sum()), "hair", int((inter&hair).sum()))
# what does the residual patch look like numerically? rows x 470..570
for cy in range(590, 711, 20):
    vals=[]
    for cx in range(460, 581, 20):
        s = final[cy-5:cy+5, cx-5:cx+5].reshape(-1,3).astype(np.float32)
        l = 0.299*s[:,0]+0.587*s[:,1]+0.114*s[:,2]; c = s.max(1)-s.min(1)
        vals.append(f"x{cx}:L{l.mean():5.0f}C{c.mean():4.0f}")
    print(f"y{cy:4d} " + " ".join(vals))
print("label ids in box rows (x460..580 step20):")
for cy in range(590, 711, 20):
    print(f"y{cy:4d} " + " ".join(f"{int(labels[cy,cx]):2d}" for cx in range(460, 581, 20)))
