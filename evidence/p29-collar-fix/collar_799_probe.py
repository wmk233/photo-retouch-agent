"""Hunt for the exact ~799 px figure the brief names."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(ROOT / "evidence/p29-neck-fix/p29_labels.png"))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(after.protected) > 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])

def band(y0, y1):
    m = np.zeros((H, W), bool); m[int(y0):int(y1), :] = True; return m

pf = labels == hp.FACE
pc = labels == np.isin(labels, hp.CLOTHING_CLASSES)
below = band(round(chin_y), H)
margin = band(round(chin_y), round(chin_y + 0.02*face_w) + 1)
below_split = band(round(chin_y + 0.02*face_w) + 0, H)

cands = {
 "A prot&below&face&~skin": prot & below & pf & ~skin,
 "B prot&margin&face&~skin": prot & margin & pf & ~skin,
 "C prot&below_split&face&~skin": prot & below_split & pf & ~skin,
 "D prot&below_split&face&~skin&pale": prot & below_split & pf & ~skin & (luma>0.9*ref_luma) & (chroma<45),
 "E prot&below_split&face&~skin&dark": prot & below_split & pf & ~skin & (luma<0.35*ref_luma),
 "F prot&margin&face&~skin&notdark": prot & margin & pf & ~skin & ~(luma<0.35*ref_luma),
 "G prot&below&face&~skin&notdark": prot & below & pf & ~skin & ~(luma<0.35*ref_luma),
 "H prot&below&~skin&allclasses": prot & below & ~skin,
 "I prot&below_split&~skin&allclasses": prot & below_split & ~skin,
}
print(json.dumps({k:int(v.sum()) for k,v in cands.items()}, indent=2))
