"""Which protected below-chin pixels are frozen by the face oval, by class."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, FACE_OVAL, _hull, build_semantic_edit_masks, fill_polygon
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(ROOT / "evidence/p29-neck-fix/p29_labels.png"))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(after.protected) > 0
collar = np.asarray(after.collar) > 0
face_core = np.asarray(after.face_core) > 0
oval = fill_polygon((H, W), _hull(lm[list(FACE_OVAL)])) > 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
below = np.zeros((H, W), bool); below[int(round(chin_y)):, :] = True

pf = labels == hp.FACE
pc = labels == hp.CLOTHING_CLASSES[0]
ph = labels == hp.HAIR
bg = labels == hp.BACKGROUND

def dil(m, ratio):
    k = max(int(round(ratio * face_w)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0

r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])

out = {}
out["chin_y"] = round(chin_y, 2)
out["oval_bottom_y"] = int(np.nonzero(oval.any(axis=1))[0].max())
for nm, cls in (("face", pf), ("cloth", pc), ("hair", ph), ("bg", bg)):
    out[f"prot_below_{nm}"] = int((prot & below & cls).sum())
    out[f"prot_below_{nm}_notskin"] = int((prot & below & cls & ~skin).sum())
out["prot_below_oval_only"] = int((prot & below & oval & ~pf & ~pc).sum())
out["prot_below_inoval_notskin"] = int((prot & below & oval & ~skin).sum())
out["prot_below_facecore"] = int((prot & below & face_core).sum())
# clothing pixels frozen: inside oval and not carved
out["prot_below_cloth_notskin_pale"] = int((prot & below & pc & ~skin & (luma > 0.9*ref_luma) & (chroma < 45)).sum())
out["prot_below_cloth_notskin_nearcollar"] = int((prot & below & pc & ~skin & dil(collar, 0.02)).sum())
# face-class pale fused to collar (the white knit above the split margin)
out["prot_below_face_notskin_pale_nearcollar"] = int((prot & below & pf & ~skin & (luma > 0.9*ref_luma) & (chroma < 45) & dil(collar, 0.02)).sum())
out["prot_below_face_notskin_dark"] = int((prot & below & pf & ~skin & (luma < 0.35*ref_luma)).sum())
out["ref_luma"] = round(ref_luma, 2)
print(json.dumps(out, indent=2))
json.dump(out, open(ROOT / "evidence/p29-collar-fix/collar_class_probe.json", "w"), indent=2)
