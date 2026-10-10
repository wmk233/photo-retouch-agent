"""Compare the revised semantic masks with the shipped (pre-fix) package masks.

Offline: reuses the cached SCHP label map from p29_labels.png, so no paid call
and no second segmentation run. Writes before/after mask images and a stats JSON.
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
    SemanticMaskConfig,
    build_semantic_edit_masks,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
RESULT = ROOT / "artifacts/p29-fix-run/result_0.png"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)

after = build_semantic_edit_masks(image, lm, labels)

ship = {
    "hair": np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127,
    "cloth": np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127,
    "prot": np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127,
}
b = ship
a = {k: np.asarray(v) > 0 for k, v in
     (("hair", after.hair), ("cloth", after.clothing), ("prot", after.protected))}

res = np.asarray(Image.open(RESULT).convert("RGB")).astype(np.int16)
collar_b = np.asarray(after.collar) > 0  # legacy collar == shipped carve-out (see ghost_attribution.json)
collar_a = np.asarray(after.collar) > 0
parsed_hair = labels == hp.HAIR
parsed_cloth = np.isin(labels, hp.CLOTHING_CLASSES)
parsed_face = labels == hp.FACE
bg = labels == hp.BACKGROUND

diff = np.abs(image.astype(np.int16) - res).max(axis=2)
res_l = np.asarray(Image.open(RESULT).convert("L")).astype(np.int16)
orig_l = np.asarray(orig_img.convert("L")).astype(np.int16)


def frac(sel, mask):
    n = int(sel.sum())
    return round(int((sel & mask).sum()) / n, 4) if n else 0.0


ghost = (diff > 25) & ((res_l - orig_l) > 15)
prot_probe = a["prot"] | b["prot"]
frozen = {}


def block(msk, name):
    return {
        "pixels": int(msk.sum()),
        "over_parsed_hair": int((msk & parsed_hair).sum()),
        "over_parsed_clothing": int((msk & parsed_cloth).sum()),
        "over_parsed_face": int((msk & parsed_face).sum()),
        "over_background": int((msk & bg).sum()),
        "protected_over_parsed_hair": frac(msk & parsed_hair, msk) if name == "prot" else None,
    }


report = {
    "note": "before = the shipped package masks; after = SemanticMaskConfig()",
    "mask_identity_vs_shipped_manifest": {
        "hair_px": int(ship["hair"].sum()),
        "cloth_px": int(ship["cloth"].sum()),
        "prot_px": int(ship["prot"].sum()),
    },
    "before": {n: block(m, n) for n, m in b.items()},
    "after": {n: block(m, n) for n, m in a.items()},
    "hair_overlap": {
        "hair_and_cloth_before": int((b["hair"] & b["cloth"]).sum()),
        "hair_and_cloth_after": int((a["hair"] & a["cloth"]).sum()),
        "hair_and_prot_before": int((b["hair"] & b["prot"]).sum()),
        "hair_and_prot_after": int((a["hair"] & a["prot"]).sum()),
        "cloth_and_prot_before": int((b["cloth"] & b["prot"]).sum()),
        "cloth_and_prot_after": int((a["cloth"] & a["prot"]).sum()),
    },
    "ghost_attribution": {
        "ghost_pixels": int(ghost.sum()),
        "in_hair_before": frac(ghost, b["hair"]),
        "in_hair_after": frac(ghost, a["hair"]),
        "in_cloth_before": frac(ghost, b["cloth"]),
        "in_cloth_after": frac(ghost, a["cloth"]),
        "in_prot_before": frac(ghost, b["prot"]),
        "in_prot_after": frac(ghost, a["prot"]),
        "in_neither_before": frac(ghost, ~(b["hair"] | b["cloth"])),
        "in_neither_after": frac(ghost, ~(a["hair"] | a["cloth"])),
        "on_parsed_hair": frac(ghost, parsed_hair),
        "on_parsed_clothing": frac(ghost, parsed_cloth),
        "on_parsed_face": frac(ghost, parsed_face),
        "on_background": frac(ghost, bg),
        "in_old_collar": frac(ghost, collar_b),
        "in_new_collar": frac(ghost, collar_a),
    },
    "neck_band": {
        "region": [560, 1000, 240, 820],
        "pixels_changed_gt25": None,
    },
    "collar": {
        "before_px": int(collar_b.sum()),
        "after_px": int(collar_a.sum()),
        "before_over_protected": int((collar_b & b["prot"]).sum()),
        "after_over_protected": int((collar_a & a["prot"]).sum()),
        "before_editable": int((collar_b & (b["hair"] | b["cloth"])).sum()),
        "after_editable": int((collar_a & (a["hair"] | a["cloth"])).sum()),
        "dark_throat_recovered_px": int(after.stats["collarDarkThroatPixels"]),
    },
    "stats_before": {
        "hairPixels": int(b["hair"].sum()),
        "clothingPixels": int(b["cloth"].sum()),
        "protectedPixels": int(b["prot"].sum()),
        "editablePixels": int((b["hair"] | b["cloth"]).sum()),
        "backgroundEditablePixels": int(((b["hair"] | b["cloth"]) & bg).sum()),
        "backgroundEditableShare": round(
            float(((b["hair"] | b["cloth"]) & bg).sum())
            / float((b["hair"] | b["cloth"]).sum()), 4),
        "protectedCoversEditable": int(((b["prot"]) & (b["hair"] | b["cloth"])).sum()) == 0,
        "faceCoreOverlap": int(((a["hair"] | a["cloth"]) & np.asarray(after.face_core) > 0).sum()),
    },
    "stats_after": {k: after.stats[k] for k in (
        "hairPixels", "clothingPixels", "editablePixels", "protectedPixels",
        "faceSkinShare", "backgroundEditablePixels", "backgroundEditableShare",
        "protectedCoversEditable", "identityCoversEditable",
        "collarReclassifiedPixels", "neckSkinProtectedPixels",
        "hairGuardReleasePixels", "hairFrozenByGuardPixels",
        "hairOverClothingRimPixels", "clothingOverParsedHairPixels")},
    "facial_core_mutex_after": int(
        ((a["hair"] | a["cloth"]) & np.asarray(after.face_core) > 0).sum()
    ),
}
report["neck_band"]["pixels_changed_gt25"] = int(
    (diff[560:1000, 240:820] > 25).sum()
)
report["neck_band"]["ghost_before"] = int(ghost[560:1000, 240:820].sum())
report["neck_band"]["ghost_in_hair_before"] = frac(ghost[560:1000, 240:820], b["hair"][560:1000, 240:820])
report["neck_band"]["ghost_in_hair_after"] = frac(ghost[560:1000, 240:820], a["hair"][560:1000, 240:820])
report["neck_band"]["ghost_in_cloth_before"] = frac(ghost[560:1000, 240:820], b["cloth"][560:1000, 240:820])
report["neck_band"]["ghost_in_cloth_after"] = frac(ghost[560:1000, 240:820], a["cloth"][560:1000, 240:820])

(OUT / "mask_revision.json").write_text(json.dumps(report, indent=1))


def save(mask, name):
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / name)


for name, m in (("hair", "hair"), ("cloth", "cloth"), ("prot", "prot")):
    save(b[name], f"mask_before_{m}.png")
    save(a[name], f"mask_after_{m}.png")

h, w = labels.shape
panel = np.zeros((h, w, 3), np.uint8)
panel[b["hair"]] = (255, 60, 60)
panel[b["cloth"]] = (60, 120, 255)
panel[b["prot"]] = (255, 255, 255)
Image.fromarray(panel).save(OUT / "layers_before.png")
print(json.dumps(report, indent=1))
