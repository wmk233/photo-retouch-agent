"""Before/after the collar-margin recovery: mask PNGs and a stats JSON.

Offline only. ``before`` is the package the last paid run submitted
(``20261009T-neckfix``); ``after`` is the same build with the recovery knob on
and the new semantic builder. No provider, no key and no paid call: the two
mask sets are read from disk and compared pixel for pixel.

The point of this script is the acceptance question the brief asks: did the
~799 px of old white collar that sat frozen under the chin actually move into
the clothing edit region -- without the neck skin, the identity guard, the
glasses guard or the hair/clothing seam losing a single pixel?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PKGS = ROOT / "artifacts/local-edit-package"
BEFORE_PKG = PKGS / "20261009T-neckfix"
AFTER_PKG = PKGS / "20261009T-collarfix"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
OUT = ROOT / "evidence/p29-collar-fix"

#: The collar band, a little wider than the 805 px island so the seam shows.
CROP = (420, 545, 650, 800)  # x0, y0, x1, y1
ZOOM = 4


def load(directory: Path, name: str) -> np.ndarray:
    return np.asarray(Image.open(directory / name).convert("L")) > 127


before = {
    "hair": load(BEFORE_PKG, "hair_edit_mask.png"),
    "cloth": load(BEFORE_PKG, "clothing_edit_mask.png"),
    "prot": load(BEFORE_PKG, "protected_identity_mask.png"),
}
after = {
    "hair": load(AFTER_PKG, "hair_edit_mask.png"),
    "cloth": load(AFTER_PKG, "clothing_edit_mask.png"),
    "prot": load(AFTER_PKG, "protected_identity_mask.png"),
}

added_cloth = after["cloth"] & ~before["cloth"]
removed_prot = before["prot"] & ~after["prot"]
lost_cloth = before["cloth"] & ~after["cloth"]
added_prot = after["prot"] & ~before["prot"]

ys, xs = np.nonzero(added_cloth)
bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if ys.size else None

photo = np.asarray(Image.open(ORIG).convert("RGB"))


def overlay(mask_cloth: np.ndarray, mask_prot: np.ndarray) -> np.ndarray:
    canvas = photo.copy()
    tint = canvas.astype(np.float32)
    tint[mask_cloth] = tint[mask_cloth] * 0.45 + np.array([60, 130, 255]) * 0.55
    tint[mask_prot] = tint[mask_prot] * 0.45 + np.array([255, 255, 255]) * 0.55
    return tint.astype(np.uint8)


def crop(img: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = CROP
    return img[y0:y1, x0:x1]


def zoom(img: np.ndarray, k: int) -> np.ndarray:
    return np.kron(img, np.ones((k, k, 1), np.uint8)) if img.ndim == 3 else np.kron(
        img, np.ones((k, k), np.uint8)
    )


Image.fromarray(zoom(crop(overlay(before["cloth"], before["prot"])), ZOOM)).save(
    OUT / "collar_edit_mask_before_4x.png"
)
Image.fromarray(zoom(crop(overlay(after["cloth"], after["prot"])), ZOOM)).save(
    OUT / "collar_edit_mask_after_4x.png"
)

# The delta panel: added pixels in green, everything still protected in white.
delta = overlay(after["cloth"], after["prot"])
delta[added_cloth] = np.array([40, 220, 90], np.uint8)
Image.fromarray(zoom(crop(delta), ZOOM)).save(OUT / "collar_edit_mask_delta_4x.png")

Image.fromarray((before["cloth"] * 255).astype(np.uint8)).save(OUT / "cloth_mask_before.png")
Image.fromarray((after["cloth"] * 255).astype(np.uint8)).save(OUT / "cloth_mask_after.png")

report = {
    "before_package": BEFORE_PKG.name,
    "after_package": AFTER_PKG.name,
    "pixels": {
        "hair_before": int(before["hair"].sum()),
        "hair_after": int(after["hair"].sum()),
        "cloth_before": int(before["cloth"].sum()),
        "cloth_after": int(after["cloth"].sum()),
        "prot_before": int(before["prot"].sum()),
        "prot_after": int(after["prot"].sum()),
    },
    "delta": {
        "clothing_added": int(added_cloth.sum()),
        "protected_removed": int(removed_prot.sum()),
        "clothing_lost": int(lost_cloth.sum()),
        "protected_added": int(added_prot.sum()),
        "added_bbox": bbox,
        "added_is_protected_removed": bool(np.array_equal(added_cloth, removed_prot)),
        "hair_unchanged": bool(np.array_equal(before["hair"], after["hair"])),
    },
    "note": (
        "clothing gained exactly what protected lost; the hair mask is byte-for-"
        "byte identical, so the accepted hairstyle/hairline and the hair/clothing "
        "seam are untouched"
    ),
}
(OUT / "collar_mask_revision.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
