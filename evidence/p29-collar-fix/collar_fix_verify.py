"""Offline before/after proof for the chin-margin collar recovery knob.

Builds the same semantic mask twice -- knob off (the last accepted mask) and
knob on -- and checks every hard gate the brief sets: neck-skin protection,
identity/spectacle guard, the hair/clothing overlap, and that the old white
collar really lands inside the clothing region.  No paid task is created.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait import edit_mask as em
from app.domains.portrait.edit_mask import SemanticMaskConfig, build_semantic_edit_masks
from app.domains.portrait.face_lock import (
    adaptive_skin_mask,
    config_from_settings,
    face_width,
    skin_reference_colour,
)
from app.services.native_mask_inpainting import NativeMaskInpainting

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-collar-fix"

image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)

cfg_off = SemanticMaskConfig(collar_margin_recovery_enabled=False)
off = build_semantic_edit_masks(image, lm, labels, config=cfg_off)
on = build_semantic_edit_masks(image, lm, labels)

face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[em.CHIN][1])
chin_row = int(round(chin_y))
below = np.zeros((H, W), bool)
below[chin_row:, :] = True

# Rebuild the two geometry guards exactly as the builder does, so the check is
# against the same masks the module used rather than against a look-alike.
shape = labels.shape
features = em.fill_feature_guard(shape, lm)
features = np.maximum(features, em.fill_polygon(shape, em._hull(lm[list(em.LIPS_INNER)])))
features = em.dilate_mask(features, cfg_off.feature_grow_ratio * face_w)
glasses = em.fill_polygon(shape, em._hull(lm[list(em.GLASSES_BAND)]))
glasses = em.dilate_mask(glasses, cfg_off.glasses_grow_ratio * face_w)
features = features > 0
glasses = glasses > 0

prot_off = np.asarray(off.protected) > 0
prot_on = np.asarray(on.protected) > 0
cloth_off = np.asarray(off.clothing) > 0
cloth_on = np.asarray(on.clothing) > 0
hair_off = np.asarray(off.hair) > 0
hair_on = np.asarray(on.hair) > 0
freed = prot_off & ~prot_on
band_off = prot_off & below & (labels == hp.FACE) & ~skin

yx = np.nonzero(freed)
freed_bbox = [int(yx[1].min()), int(yx[0].min()), int(yx[1].max()), int(yx[0].max())] if len(yx[0]) else None

report = {
    "knob": {
        "enabled_default": SemanticMaskConfig().collar_margin_recovery_enabled,
        "luma_ratio": SemanticMaskConfig().collar_margin_recovery_luma_ratio,
        "chroma_ceiling": SemanticMaskConfig().collar_margin_recovery_chroma,
        "ring_ratio": SemanticMaskConfig().collar_margin_recovery_ring_ratio,
        "ring_px": round(SemanticMaskConfig().collar_margin_recovery_ring_ratio * face_w, 2),
    },
    "chin_row": chin_row,
    "the_805_band": int(band_off.sum()),
    "freed_pixels": int(freed.sum()),
    "freed_bbox": freed_bbox,
    "hard_gates": {
        "freed_over_skin": int((freed & skin).sum()),
        "freed_above_chin_row": int((freed & ~below).sum()),
        "freed_on_parsed_hair": int((freed & (labels == hp.HAIR)).sum()),
        "freed_on_identity_guard": int((freed & features).sum()),
        "freed_on_glasses_guard": int((freed & glasses).sum()),
        "neck_skin_protected_off": int((prot_off & skin).sum()),
        "neck_skin_protected_on": int((prot_on & skin).sum()),
        "identity_guard_protected_off": int((prot_off & features).sum()),
        "identity_guard_protected_on": int((prot_on & features).sum()),
        "glasses_guard_protected_off": int((prot_off & glasses).sum()),
        "glasses_guard_protected_on": int((prot_on & glasses).sum()),
        "hair_frozen_by_guard_off": int((prot_off & (labels == hp.HAIR)).sum()),
        "hair_frozen_by_guard_on": int((prot_on & (labels == hp.HAIR)).sum()),
        "hair_clothing_overlap_off": int((hair_off & cloth_off).sum()),
        "hair_clothing_overlap_on": int((hair_on & cloth_on).sum()),
        "protected_total_off": int(prot_off.sum()),
        "protected_total_on": int(prot_on.sum()),
        "editable_total_off": int((np.asarray(off.editable) > 0).sum()),
        "editable_total_on": int((np.asarray(on.editable) > 0).sum()),
    },
    "recovered_now_inside_clothing": int((freed & cloth_on).sum()),
    "recovered_now_editable": int((freed & (np.asarray(on.editable) > 0)).sum()),
    "band_freed": int((band_off & freed).sum()),
    "band_still_protected": int((band_off & prot_on).sum()),
    "band_still_protected_dark": int(
        (band_off & prot_on & (
            em._luma(image) < cfg_off.collar_skin_luma_ratio
            * float(em._luma(np.asarray(reference, np.float32).reshape(1, 1, 3))[0, 0])
        )).sum()
    ),
    "ons_stats": {
        k: v for k, v in on.stats.items() if k.startswith("collarMargin")
    },
    "off_stats": {
        k: off.stats[k]
        for k in ("protectedPixels", "hairPixels", "clothingPixels", "editablePixels")
    },
    "on_stats": {
        k: on.stats[k] for k in ("protectedPixels", "clothingPixels", "editablePixels")
    },
    "gaps": list(on.gaps),
}

(OUT / "collar_fix_verify.json").write_text(json.dumps(report, indent=2))

# Before/after mask picture over the original, 8x, same crop as the band views.
ov = image.copy()
vis_band = band_off & prot_on
ov[vis_band] = (0.35 * ov[vis_band] + 0.65 * np.array([190, 0, 0])).astype(np.uint8)
ov[freed] = np.array([0, 255, 255], np.uint8)
Image.fromarray(ov[560:660, 360:600]).resize((240 * 8, 100 * 8), Image.NEAREST).save(
    OUT / "collar_recovery_after_8x.png"
)
Image.fromarray(ov[560:660, 360:600]).save(OUT / "collar_recovery_after_1x.png")
print(json.dumps({k: report[k] for k in ("the_805_band", "freed_pixels", "freed_bbox")}, indent=2))
print(json.dumps(report["hard_gates"], indent=2))
print("recovered_now_inside_clothing", report["recovered_now_inside_clothing"])
print("band_freed", report["band_freed"], "band_still_protected", report["band_still_protected"], "dark", report["band_still_protected_dark"])
print("off_stats", report["off_stats"])
