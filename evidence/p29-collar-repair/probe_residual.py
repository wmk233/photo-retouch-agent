"""Probe: where is the pale collar residual in the BEST final (task 2108330225116082177)?

Offline. Reports the residual's extent, its class context from the SCHP parse of
the original, and how much of it collides with the identity/skin guards.
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
from app.domains.portrait.edit_mask import CHIN, FACE_OVAL, _hull, build_semantic_edit_masks, fill_polygon  # noqa: E402
from app.domains.portrait.face_lock import (  # noqa: E402
    adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
FINAL = ROOT / "evidence/p29-submit-run/final.png"
LABELS = ROOT / "evidence/p29-collar-repair/labels_fresh.png"
OUT = ROOT / "evidence/p29-collar-repair"

image = np.asarray(Image.open(ORIG).convert("RGB"))
final = np.asarray(Image.open(FINAL).convert("RGB"))
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
assert final.shape[:2] == (H, W), final.shape

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
ms = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(ms.protected) > 0
hair = np.asarray(ms.hair) > 0
cloth = np.asarray(ms.clothing) > 0
collar = np.asarray(ms.collar) > 0
parsed_face = np.asarray(ms.parsed_face) > 0
face_core = np.asarray(ms.face_core) > 0
bg = np.isin(labels, [0])  # ATR Background
face_w = face_width(lm)
chin_y = float(lm[CHIN][1])
print(f"face_w={face_w:.1f} chin_y={chin_y:.1f}")

reference, _ = skin_reference_colour(image, lm, face_w)
lock_cfg = config_from_settings(settings)
skin = adaptive_skin_mask(image, reference, face_w, lock_cfg) > 0
oval = fill_polygon((H, W), _hull(lm[list(FACE_OVAL)])) > 0

def luma_chroma(img):
    a = img.astype(np.float32)
    luma = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    chroma = a.max(2) - a.min(2)
    return luma, chroma

fl, fc = luma_chroma(final)
ol, oc = luma_chroma(image)

pale = (fl > 165) & (fc < 45)
orig_pale = (ol > 165) & (oc < 45)

# blouse reference: sample the final in the correct dark-olive body (below chin, centre-bottom sides)
body = np.zeros((H, W), bool)
body[int(chin_y) + 120 :, :] = True
body &= ~prot & ~hair
print("final garment-body mean RGB:", final[body].mean(0).round(1))

report = {
    "face_w": round(face_w, 2),
    "chin_y": round(chin_y, 2),
    "final_pale_px": int(pale.sum()),
    "orig_pale_px": int(orig_pale.sum()),
    "pale_in_cloth": int((pale & cloth).sum()),
    "pale_in_protected": int((pale & prot).sum()),
    "pale_in_hair": int((pale & hair).sum()),
    "pale_in_background": int((pale & bg).sum()),
    "pale_in_collar": int((pale & collar).sum()),
    "pale_in_skin": int((pale & skin).sum()),
    "pale_in_oval": int((pale & oval).sum()),
    "orig_pale_in_protected": int((orig_pale & prot).sum()),
    "mask_px": {"hair": int(hair.sum()), "cloth": int(cloth.sum()), "prot": int(prot.sum()),
                "collar": int(collar.sum()), "parsed_face": int(parsed_face.sum()), "face_core": int(face_core.sum())},
}

# Rough bounding box of the pale residual that sits over garment/protected collar area
sel = pale & (cloth | collar | prot)
ys, xs = np.where(sel)
if len(ys):
    report["pale_over_garment_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    report["pale_over_garment_px"] = int(len(ys))

# Rows histogram of pale over garment column band
sub = pale & cloth
rows = sub.sum(1)
nz = np.where(rows > 0)[0]
if len(nz):
    report["pale_over_cloth_rows"] = [int(nz.min()), int(nz.max())]
cols = sub.sum(0)
nzc = np.where(cols > 0)[0]
if len(nzc):
    report["pale_over_cloth_cols"] = [int(nzc.min()), int(nzc.max())]

print(json.dumps(report, indent=1))
json.dump(report, open(OUT / "probe_residual.json", "w"), indent=1)

# preview overlay
vis = final.copy()
vis[pale & cloth] = [255, 0, 0]
vis[pale & prot] = [0, 255, 255]
vis[collar & ~pale] = [0, 0, 255]
vis[hair & pale] = [255, 0, 255]
Image.fromarray(vis).save(OUT / "probe_residual_overlay.png")
print("wrote probe_residual_overlay.png")
