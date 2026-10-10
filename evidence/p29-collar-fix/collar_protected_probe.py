"""Locate old white collar pixels that are still inside the protected guard.

Offline: reuses the cached SCHP label map (evidence/p29-neck-fix/p29_labels.png)
and the shipped edit-mask package, so no paid call and no re-segmentation. The
brief for this round names ~799 px of old white collar that the guard is still
freezing; this probe measures candidates under a range of evidence rules before
any knob is introduced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.edit_mask import (  # noqa: E402
    build_semantic_edit_masks,
)
from app.domains.portrait.face_lock import (  # noqa: E402
    adaptive_skin_mask,
    config_from_settings,
    face_width,
    skin_reference_colour,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-collar-fix"

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)

prot = np.asarray(after.protected) > 0
cloth = np.asarray(after.clothing) > 0
hair = np.asarray(after.hair) > 0
collar = np.asarray(after.collar) > 0
face_core = np.asarray(after.face_core) > 0

ship_prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127
ship_cloth = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
ship_hair = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127

r = image[:, :, 0].astype(np.float32)
g = image[:, :, 1].astype(np.float32)
b = image[:, :, 2].astype(np.float32)
luma = 0.299 * r + 0.587 * g + 0.114 * b
mx = np.maximum(np.maximum(r, g), b)
mn = np.minimum(np.minimum(r, g), b)
chroma = mx - mn

face_w = face_width(lm)
reference, sample_px = skin_reference_colour(image, lm, face_w)
lock_cfg = config_from_settings(settings)
skin = adaptive_skin_mask(image, reference, face_w, lock_cfg) > 0
print("face_w", round(face_w, 1), "skin ref", None if reference is None else reference.round(1).tolist(),
      "sample_px", sample_px)

chin_y = float(lm[hp.CHIN][1]) if hasattr(hp, "CHIN") else None
# use the same landmark index the mask builder uses
from app.domains.portrait.edit_mask import CHIN  # noqa: E402
chin_y = float(lm[CHIN][1])

parsed_face = labels == hp.FACE
parsed_cloth = np.isin(labels, hp.CLOTHING_CLASSES)
parsed_hair = labels == hp.HAIR
bg = labels == hp.BACKGROUND

import cv2  # noqa: E402

def dil(m, ratio):
    k = max(int(round(ratio * face_w)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0

band = np.zeros((H, W), bool)
band[max(int(round(chin_y - 0.10 * face_w)), 0):, :] = True

report = {}
report["faceWidthPx"] = round(face_w, 2)
report["chinY"] = round(chin_y, 2)
report["shipProtectedPixels"] = int(ship_prot.sum())
report["afterProtectedPixels"] = int(prot.sum())
report["shipClothingPixels"] = int(ship_cloth.sum())
report["afterClothingPixels"] = int(cloth.sum())

# ---- candidate families -------------------------------------------------
pale = (luma > 165) & (chroma < 42)
near_cloth = dil(parsed_cloth, 0.05)
near_collar = dil(collar, 0.02)
adj_cloth = dil(parsed_cloth, 0.02)

fam = {}
def add(name, m):
    fam[name] = {
        "pixels": int(m.sum()),
        "in_protected_after": int((m & prot).sum()),
        "in_protected_ship": int((m & ship_prot).sum()),
        "not_skin": int((m & ~skin).sum()),
        "in_band": int((m & band).sum()),
    }

below = np.zeros((H, W), bool)
below[max(int(round(chin_y)), 0):, :] = True

add("pale_all", pale)
add("pale_protected", pale & prot)
add("pale_protected_belowchin", pale & prot & below)
add("pale_protected_nearcloth", pale & prot & near_cloth)
add("pale_protected_nearcloth_notskin", pale & prot & near_cloth & ~skin)
add("pale_protected_band_notskin", pale & prot & band & ~skin)
add("pale_protected_belowchin_notskin", pale & prot & below & ~skin)
add("pale_notskin_nearcloth_nearcollar", (pale & ~skin & near_cloth & near_collar))
add("prot_notskin_nearcloth", prot & ~skin & near_cloth)
add("shipprot_notskin_nearcloth", ship_prot & ~skin & near_cloth)
add("prot_pale_belowchin_adjcloth", pale & prot & below & dil(parsed_cloth, 0.01))
add("parsedface_prot_notskin_below", parsed_face & prot & below & ~skin)
add("collar_pale", collar & pale)
add("parsedcloth_pale", parsed_cloth & pale)
add("prot_pale_where_parsedcloth", pale & prot & parsed_cloth)
add("prot_pale_where_shipcloth", pale & prot & ship_cloth)

report["families"] = fam
report["bands"] = {
    "paleThresholds": {
        f"luma>{t}_chroma<{c}": int(((luma > t) & (chroma < c) & prot).sum())
        for t in (150, 160, 170, 180) for c in (30, 42, 55)
    }
}

OUT.joinpath("collar_protected_probe.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
