"""Profile the below-chin, non-skin, protected parsed-Face pixels (the ~799 px).

Offline. Builds an evidence table (luma/chroma/SCHP class/distance to the carved
collar and to the parsed garment) for every candidate pixel, and writes a debug
overlay so the decision can be made on sight.
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
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks  # noqa: E402
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

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)

prot = np.asarray(after.protected) > 0
collar = np.asarray(after.collar) > 0
face_core = np.asarray(after.face_core) > 0
parsed_face = np.asarray(after.parsed_face) > 0

face_w = face_width(lm)
reference, sample_px = skin_reference_colour(image, lm, face_w)
lock_cfg = config_from_settings(settings)
skin = adaptive_skin_mask(image, reference, face_w, lock_cfg) > 0
chin_y = float(lm[CHIN][1])

r = image[:, :, 0].astype(np.float32)
g = image[:, :, 1].astype(np.float32)
b = image[:, :, 2].astype(np.float32)
luma = 0.299 * r + 0.587 * g + 0.114 * b
chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

ref_luma = float(0.299 * reference[0] + 0.587 * reference[1] + 0.114 * reference[2])
below = np.zeros((H, W), bool)
below[int(round(chin_y)):, :] = True

cand = parsed_face & prot & below & ~skin
ys, xs = np.nonzero(cand)
print("candidate pixels:", len(ys))
if len(ys):
    print("bbox x", xs.min(), xs.max(), "y", ys.min(), ys.max())
    print("chin_y", round(chin_y, 2), "face_w", round(face_w, 2), "ref_luma", round(ref_luma, 2))
    print("luma: min/med/max", round(float(luma[cand].min()), 1), round(float(np.median(luma[cand])), 1), round(float(luma[cand].max()), 1))
    print("chroma: min/med/max", round(float(chroma[cand].min()), 1), round(float(np.median(chroma[cand])), 1), round(float(chroma[cand].max()), 1))
    print("luma/ref_luma: min/med/max", round(float((luma[cand]/ref_luma).min()), 2), round(float(np.median(luma[cand]/ref_luma)), 2), round(float((luma[cand]/ref_luma).max()), 2))
    print("parsed_cloth in cand:", int((cand & (labels == hp.BACKGROUND)).sum()))
    print("cand on parsed_bg:", int((cand & (labels == hp.BACKGROUND)).sum()))
    print("cand on parsed_cloth:", int((cand & np.isin(labels, hp.CLOTHING_CLASSES)).sum()))
    print("cand on parsed_hair:", int((cand & (labels == hp.HAIR)).sum()))

def dil(m, ratio):
    k = max(int(round(ratio * face_w)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0

rows = []
for ratio in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
    rows.append({
        "ratio": ratio,
        "px": round(ratio * face_w, 1),
        "near_collar": int((cand & dil(collar, ratio)).sum()),
        "near_parsed_cloth": int((cand & dil(np.isin(labels, hp.CLOTHING_CLASSES), ratio)).sum()),
        "near_face_core": int((cand & dil(face_core, ratio)).sum()),
    })
print(json.dumps(rows, indent=2))

# connected components of the candidate
n, lab, stats, cent = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
comps = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    comps.append({"area": int(area), "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                  "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1)})
comps.sort(key=lambda c: -c["area"])
print("components:", json.dumps(comps[:12], indent=2))

# pale subset (white knit): brighter than skin, low chroma
pale = (luma > reference.mean()) & (chroma < 45)
pale_cand = cand & pale
print("cand pale px:", int(pale_cand.sum()), "cand dark px:", int((cand & (luma < 0.35 * ref_luma)).sum()))

# debug overlay
dbg = image.copy()
dbg[cand] = [255, 0, 0]
dbg[pale_cand] = [255, 255, 0]
Image.fromarray(dbg).save(OUT / "collar_protected_evidence_overlay.png")
y0, y1 = max(int(chin_y) - 60, 0), min(int(chin_y) + 220, H)
x0, x1 = 300, 640
crop = dbg[y0:y1, x0:x1]
Image.fromarray(crop).resize(((x1 - x0) * 4, (y1 - y0) * 4), Image.NEAREST).save(OUT / "collar_protected_evidence_zoom4x.png")
print("wrote overlay + zoom4x", (x0, y0, x1, y1))

json.dump({"candidate_pixels": int(cand.sum()), "pale_candidate_pixels": int(pale_cand.sum()),
           "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None,
           "proximity": rows, "components": comps[:20]},
          open(OUT / "collar_protected_evidence.json", "w"), indent=2)
