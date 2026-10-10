"""Evaluate the proposed narrow recovery rule and prove it cannot touch skin/features."""
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
collar = np.asarray(after.collar) > 0
pf = labels == hp.FACE
pc = np.isin(labels, hp.CLOTHING_CLASSES)
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])
below = np.zeros((H, W), bool); below[int(round(chin_y)):, :] = True

def dil(m, ratio):
    k = max(int(round(ratio * face_w)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0

ring = dil(collar | pc, 0.02)
notskin = ~skin
nondark = luma >= 0.35 * ref_luma
desat = chroma < 45

rule = prot & below & notskin & nondark & ring & desat
rule_cloth = prot & below & notskin & nondark & ring & pc  # parser says garment
union = rule | rule_cloth

def n(m): return int(m.sum())
out = {
 "protected_total": n(prot),
 "target_band(prot&below&face&~skin)": n(prot & below & pf & notskin),
 "recover_desat_ring": n(rule),
 "recover_schp_cloth_ring": n(rule_cloth),
 "recover_union": n(union),
 "recover_union_pale_only(luma>0.9ref)": n(union & (luma > 0.9*ref_luma)),
 "recover_union_dark(<0.35ref)": n(union & (luma < 0.35*ref_luma)),
 "recover_union_over_skin": n(union & skin),
 "recover_union_above_chin": n(union & ~below),
 "recover_union_on_hair": n(union & (labels == hp.HAIR)),
 "remaining_target_after": n(prot & below & pf & notskin & ~union),
 "ring_ratio_px": round(0.02*face_w, 2),
}
print(json.dumps(out, indent=2))

# bbox of the union
ys, xs = np.nonzero(union)
out["union_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None
print("union bbox", out["union_bbox"])

ov = np.asarray(Image.open(ORIG).convert("RGB")).copy()
ov[prot & below & pf & notskin] = (0.5*ov[prot & below & pf & notskin] + 0.5*np.array([255,0,0])).astype(np.uint8)
ov[union] = np.array([0, 255, 255], np.uint8)
Image.fromarray(ov[560:640, 360:600]).resize((240*8, 80*8), Image.NEAREST).save(ROOT/"evidence/p29-collar-fix/recovery_rule_zoom8x.png")
json.dump(out, open(ROOT/"evidence/p29-collar-fix/collar_recovery_rule.json","w"), indent=2)
