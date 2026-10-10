"""Pick the mask post-processing rules from measurements, not guesses.

Offline (labels already on disk, no provider call):
  * how much below-chin Face is shadowed throat vs a real collar
  * how the hair mask's spill onto parsed clothing decomposes by distance
  * how much of the protected guard is the subject's own parsed hair
"""
from __future__ import annotations

import dataclasses
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
from app.domains.portrait.face_lock import (  # noqa: E402
    CHIN,
    FACE_OVAL,
    adaptive_skin_mask,
    dilate_mask,
    fill_polygon,
    skin_reference_colour,
)
from app.domains.portrait.edit_mask import build_semantic_edit_masks  # noqa: E402
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
mset = build_semantic_edit_masks(image, lm, labels)
hair_m = np.asarray(mset.hair) > 0
cloth_m = np.asarray(mset.clothing) > 0
prot_m = np.asarray(mset.protected) > 0

hair_class = labels == hp.HAIR
cloth_class = np.isin(labels, hp.CLOTHING_CLASSES)
face_class = labels == hp.FACE
upper_class = labels == hp.UPPER_CLOTHES
bg_class = labels == hp.BACKGROUND

cfg = settings
from app.domains.portrait.face_lock import config_from_settings  # noqa: E402

lock_cfg = config_from_settings(settings)
face_w = mset.face_width_px
chin_y = float(lm[CHIN][1])

ref, sample_px = skin_reference_colour(image, lm, face_w)
report: dict = {"face_width_px": round(face_w, 3), "chin_y": round(chin_y, 2),
                "skin_sample_px": sample_px}

# ---------- A. below-chin Face: shadowed throat vs real collar ----------
below = np.zeros((H, W), bool)
below[int(round(chin_y)):, :] = True
below_face = face_class & below
oval = fill_polygon((H, W), lm[list(FACE_OVAL)]) > 0

report["below_chin_face_px"] = int(below_face.sum())
tone_rows = []
for y_ratio in (0.65, 0.55, 0.45, 0.35, 0.25, 0.15):
    c = dataclasses.replace(lock_cfg, skin_y_ratio=y_ratio)
    sm = adaptive_skin_mask(image, ref, face_w, c) > 0
    skin_px = int((below_face & sm).sum())
    collar_px = int((below_face & ~sm).sum())
    tone_rows.append({"skin_y_ratio": y_ratio, "skin_px": skin_px,
                      "collar_px": collar_px})
report["below_chin_tone_sweep"] = tone_rows

# chroma of the shadowed part: how far the current "collar" is from skin tone
c065 = dataclasses.replace(lock_cfg, skin_y_ratio=0.65)
sm065 = adaptive_skin_mask(image, ref, face_w, c065) > 0
collar_now = below_face & ~sm065
lum = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)
ref_lum = float(np.dot(np.asarray(ref, np.float32), [0.299, 0.587, 0.114]))
report["collar_now_px"] = int(collar_now.sum())
if collar_now.any():
    l = lum[collar_now]
    px = image[collar_now].astype(np.float32)
    report["collar_now_luma"] = {
        "mean": round(float(l.mean()), 2),
        "p05": round(float(np.percentile(l, 5)), 2),
        "p50": round(float(np.percentile(l, 50)), 2),
        "ref_lum": round(ref_lum, 2),
        "share_above_0.35ref": round(float((l > 0.35 * ref_lum).mean()), 4),
        "share_above_0.65ref": round(float((l > 0.65 * ref_lum).mean()), 4),
    }
    # distance from reference skin in RGB (is it the subject's own skin, shadowed?)
    d = np.linalg.norm(px - np.asarray(ref, np.float32), axis=1)
    report["collar_now_rgb_distance_to_skin"] = {
        "mean": round(float(d.mean()), 2),
        "p50": round(float(np.percentile(d, 50)), 2),
        "share_within_60": round(float((d < 60).mean()), 4),
    }
    # how much of the current collar is inside the oval (chin is low) vs below
    report["collar_now_over_oval_px"] = int((collar_now & oval).sum())
    report["collar_now_below_oval_px"] = int((collar_now & ~oval).sum())

# ---------- B. hair mask spill onto clothing ----------
spill = hair_m & cloth_class
report["hair_mask_px"] = int(hair_m.sum())
report["hair_mask_over_clothing_px"] = int(spill.sum())
report["hair_mask_over_hair_px"] = int((hair_m & hair_class).sum())
report["hair_mask_over_bg_px"] = int((hair_m & bg_class).sum())
report["hair_mask_over_face_px"] = int((hair_m & face_class).sum())

# distance from the parsed hair silhouette: is the spill a thin rim or deep?
ha = hair_class.astype(np.uint8)
dist = cv2.distanceTransform(1 - ha, cv2.DIST_L2, 3)
d_spill = dist[spill]
if spill.any():
    report["spill_distance_from_parsed_hair_px"] = {
        f"share_le_{k}": round(float((d_spill <= k).mean()), 4)
        for k in (2, 5, 8, 14, 20, 31)
    }
    report["spill_max_distance_px"] = round(float(d_spill.max()), 2)
    report["spill_p50_distance_px"] = round(float(np.percentile(d_spill, 50)), 2)

# proposed rim rules, arithmetic only (no mask re-render yet)
for tight in (0.02, 0.04):
    rim_full = dilate_mask(hair_class.astype(np.uint8) * 255, 0.12 * face_w)
    rim_tight = dilate_mask(hair_class.astype(np.uint8) * 255, tight * face_w)
    cand = np.maximum(hair_class.astype(np.uint8) * 255,
                      np.maximum(cv2.bitwise_and(rim_full, cv2.bitwise_not(cloth_class.astype(np.uint8) * 255)),
                                 cv2.bitwise_and(rim_tight, cloth_class.astype(np.uint8) * 255)))
    cand = cv2.bitwise_and(cand, cv2.bitwise_not(prot_m.astype(np.uint8) * 255)) > 0
    report[f"rule_hair_over_clothing_rim_{tight}"] = {
        "hair_px": int(cand.sum()),
        "over_clothing_px": int((cand & cloth_class).sum()),
        "over_bg_px": int((cand & bg_class).sum()),
        "over_face_px": int((cand & face_class).sum()),
        "over_protected_px": int((cand & prot_m).sum()),
    }

# ---------- C. protected guard over the subject's own hair ----------
frozen = prot_m & hair_class
report["protected_px"] = int(prot_m.sum())
report["protected_over_parsed_hair_px"] = int(frozen.sum())
report["protected_over_parsed_face_px"] = int((prot_m & face_class).sum())
report["protected_over_bg_px"] = int((prot_m & bg_class).sum())

# and where the fringe is: rows
ys = np.where(frozen.any(axis=1))[0]
if ys.size:
    report["frozen_hair_rows"] = [int(ys.min()), int(ys.max())]

(OUT / "rule_probe.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
