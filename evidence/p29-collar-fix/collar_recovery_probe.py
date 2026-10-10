"""Numeric probe: which protected below-chin pixels look like the old white collar.

Offline. No knob is applied here; the goal is to choose the narrowest rule that
frees genuinely garment pixels and to prove the skin/feature guards survive it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.edit_mask import (  # noqa: E402
    CHIN,
    FACE_OVAL,
    _hull,
    build_semantic_edit_masks,
    fill_polygon,
)
from app.domains.portrait.face_lock import (  # noqa: E402
    adaptive_skin_mask,
    config_from_settings,
    face_width,
    skin_reference_colour,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-collar-fix"

image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)

prot = np.asarray(after.protected) > 0
collar = np.asarray(after.collar) > 0
parsed_face = np.asarray(after.parsed_face) > 0
parsed_cloth = np.isin(labels, hp.CLOTHING_CLASSES)
oval = fill_polygon((H, W), _hull(lm[list(FACE_OVAL)])) > 0

face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
lock_cfg = config_from_settings(settings)
skin = adaptive_skin_mask(image, reference, face_w, lock_cfg) > 0
chin_y = float(lm[CHIN][1])

r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
luma = 0.299 * r + 0.587 * g + 0.114 * b
chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
ref_luma = float(0.299 * reference[0] + 0.587 * reference[1] + 0.114 * reference[2])

def band(y0, y1):
    m = np.zeros((H, W), bool)
    m[max(int(y0), 0):int(y1), :] = True
    return m

below = band(round(chin_y), H)
margin_band = band(round(chin_y), round(chin_y + 0.02 * face_w) + 1)

def dil(m, ratio):
    k = max(int(round(ratio * face_w)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0

near_collar = dil(collar, 0.02)
near_cloth = dil(parsed_cloth, 0.02)

def count(m):
    return int(m.sum())

res = {}
res["protected_total"] = count(prot)
res["belowychin_notskin"] = count(prot & below & ~skin)
res["margin_band_notskin"] = count(prot & margin_band & ~skin)
res["belowychin_parsedface_notskin"] = count(prot & below & parsed_face & ~skin)
res["belowychin_parsedcloth"] = count(prot & below & parsed_cloth)
res["belowychin_oval_only_notskin"] = count(prot & below & oval & ~parsed_face & ~skin)
res["belowychin_allclasses_notskin"] = count(prot & below & ~skin)

pale_defs = {
    "luma>0.9refluma_chroma<45": (luma > 0.9 * ref_luma) & (chroma < 45),
    "luma>1.0refluma_chroma<45": (luma > 1.0 * ref_luma) & (chroma < 45),
    "luma>1.1refluma_chroma<45": (luma > 1.1 * ref_luma) & (chroma < 45),
    "luma>0.9refluma_chroma<60": (luma > 0.9 * ref_luma) & (chroma < 60),
}
for name, pale in pale_defs.items():
    res[f"pale[{name}]_belowychin_notskin"] = count(prot & below & ~skin & pale)
    res[f"pale[{name}]_belowychin_notskin_nearcollar"] = count(prot & below & ~skin & pale & near_collar)
    res[f"pale[{name}]_belowychin_notskin_nearcloth"] = count(prot & below & ~skin & pale & near_cloth)

# the strict, well-evidenced rule: classified clothing OR (non-skin AND pale AND fused to the carried collar)
strict = prot & below & parsed_cloth & ~skin
strict2 = prot & below & ~skin & (luma > 0.9 * ref_luma) & (chroma < 45) & dil(collar, 0.02)
res["strict_schp_clothing_belowychin"] = count(strict)
res["strict_pale_fused_collar"] = count(strict2)
res["strict_union"] = count(strict | strict2)
res["strict_union_pale_only"] = count((strict | strict2) & (luma > 0.9 * ref_luma) & (chroma < 45))

# safety: none of these may touch features/glasses
feat = np.asarray(after.protected) > 0  # placeholder
res["strict_union_in_oval"] = count((strict | strict2) & oval)
res["strict_union_minus_skin"] = count((strict | strict2) & skin)

print(json.dumps(res, indent=2))
json.dump(res, open(OUT / "collar_recovery_probe.json", "w"), indent=2)
