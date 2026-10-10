"""Attribute the chin/neck/chest ghost to masks AND to the real ATR parse.

Offline: one local SCHP inference (free), the two real RunningHub results and
the shipped masks. No provider call, no spend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.face_lock import CHIN  # noqa: E402
from app.domains.portrait.edit_mask import build_semantic_edit_masks  # noqa: E402
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
RESULT = ROOT / "artifacts/p29-fix-run/result_0.png"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
res = np.asarray(Image.open(RESULT).convert("RGB")).astype(np.int16)
orig = image.astype(np.int16)
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape

hair_class = labels == hp.HAIR
upper_class = labels == hp.UPPER_CLOTHES
cloth_class = np.isin(labels, hp.CLOTHING_CLASSES)
face_class = labels == hp.FACE
bg_class = labels == hp.BACKGROUND

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
mask_set = build_semantic_edit_masks(image, lm, labels)
hair_m = np.asarray(mask_set.hair) > 0
cloth_m = np.asarray(mask_set.clothing) > 0
prot_m = np.asarray(mask_set.protected) > 0
collar_m = np.asarray(mask_set.collar) > 0

# shipped package masks, for the "before" comparison the report quotes
ship_hair = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127
ship_cloth = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
ship_prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127

chin_y = int(round(float(lm[CHIN][1])))

diff = np.abs(orig - res).max(axis=2)
lum_delta = np.asarray(Image.open(RESULT).convert("L")).astype(np.int16) - np.asarray(
    orig_img.convert("L")
).astype(np.int16)

# the ghost region as the user describes it: chin -> neck -> upper chest
region = np.zeros((H, W), bool)
region[560:1000, 240:820] = True


def split(sel: np.ndarray) -> dict:
    n = int(sel.sum())
    if n == 0:
        return {"pixels": 0}

    def frac(mask: np.ndarray) -> float:
        return round(int((sel & mask).sum()) / n, 4)

    return {
        "pixels": n,
        "parse_hair": frac(hair_class),
        "parse_upper_clothes": frac(upper_class),
        "parse_any_clothing": frac(cloth_class),
        "parse_face_or_neck": frac(face_class),
        "parse_background": frac(bg_class),
        "mask_hair": frac(hair_m),
        "mask_clothing": frac(cloth_m),
        "mask_protected": frac(prot_m),
        "mask_both_edit": frac(hair_m & cloth_m),
        "mean_abs_delta": round(float(diff[sel].mean()), 2),
        "identical_le2": round(float((diff[sel] <= 2).mean()), 4),
        "brighter_than_original": round(float((lum_delta[sel] > 15).mean()), 4),
    }


report: dict = {"chin_y": chin_y}

ghost = region & (diff > 25) & (lum_delta > 15)
report["ghost_bright_rule"] = "diff>25 and result >15 luminance brighter (the pale ghost)"
report["ghost_bright"] = split(ghost)

hard = region & (diff > 40)
report["region_changed_le40"] = split(region & (diff > 40))

# the old pale knit collar: near-white pixels in the chest region
near_white = (orig.min(axis=2) > 160) & (orig.max(axis=2) > 195)
old_collar = region & near_white
report["old_pale_knit_pixels"] = int(old_collar.sum())
report["old_pale_knit"] = split(old_collar)
report["old_pale_knit_matched_parse_upper"] = round(
    float((old_collar & upper_class).sum() / max(int(old_collar.sum()), 1)), 4
)

# neck skin per the parse: Face pixels below the chin row
if chin_y is not None:
    below = np.zeros((H, W), bool)
    below[chin_y:, :] = True
    neck = region & below & face_class
    report["neck_parse_face_pixels"] = int(neck.sum())
    report["neck_parse_face"] = split(neck)

# hair mask over the garment -------------------------------------------------
report["hair_mask_over_parsed_clothing_px"] = int((hair_m & cloth_class).sum())
report["hair_mask_over_parsed_hair_px"] = int((hair_m & hair_class).sum())
report["hair_mask_over_parsed_bg_px"] = int((hair_m & bg_class).sum())
report["shipped_hair_over_parsed_clothing_px"] = int((ship_hair & cloth_class).sum())
report["shipped_hair_over_parsed_bg_px"] = int((ship_hair & bg_class).sum())
report["cloth_mask_over_parsed_hair_px"] = int((cloth_m & hair_class).sum())
report["cloth_mask_over_parsed_clothing_px"] = int((cloth_m & cloth_class).sum())
report["cloth_mask_over_parsed_face_px"] = int((cloth_m & face_class).sum())
report["shipped_cloth_over_parsed_face_px"] = int((ship_cloth & face_class).sum())
report["collar_mask_px"] = int(collar_m.sum())
report["collar_mask_over_protected_px"] = int((collar_m & prot_m).sum())
report["shipped_hair_equals_rebuilt_px"] = int((ship_hair ^ hair_m).sum())
report["shipped_cloth_equals_rebuilt_px"] = int((ship_cloth ^ cloth_m).sum())
report["shipped_prot_equals_rebuilt_px"] = int((ship_prot ^ prot_m).sum())

# where does the ghost sit relative to the edit masks?
ghost_bright = ghost
report["ghost_in_hair_only"] = int((ghost_bright & hair_m & ~cloth_m).sum())
report["ghost_in_cloth_only"] = int((ghost_bright & cloth_m & ~hair_m).sum())
report["ghost_in_both"] = int((ghost_bright & hair_m & cloth_m).sum())
report["ghost_in_neither"] = int((ghost_bright & ~hair_m & ~cloth_m).sum())
report["ghost_on_parsed_clothing"] = int((ghost_bright & cloth_class).sum())
report["ghost_on_parsed_hair"] = int((ghost_bright & hair_class).sum())
report["ghost_on_parsed_face"] = int((ghost_bright & face_class).sum())

(OUT / "ghost_attribution.json").write_text(json.dumps(report, indent=1))

# ------------------------------------------------------------------- figure
BOX = (240, 520, 840, 1010)
x0, y0, x1, y1 = BOX
palette = {
    "hair": (255, 150, 40),
    "upper": (70, 130, 255),
    "other_cloth": (150, 110, 255),
    "face": (60, 220, 90),
    "bg": (30, 30, 40),
}
cls_rgb = np.zeros((H, W, 3), np.uint8)
cls_rgb[bg_class] = palette["bg"]
cls_rgb[hair_class] = palette["hair"]
cls_rgb[upper_class] = palette["upper"]
cls_rgb[cloth_class & ~upper_class] = palette["other_cloth"]
cls_rgb[face_class] = palette["face"]

own = np.asarray(orig_img).copy()
own[hair_m] = (own[hair_m] * 0.35 + np.array([255, 60, 60]) * 0.65).astype(np.uint8)
own[cloth_m] = (own[cloth_m] * 0.35 + np.array([60, 110, 255]) * 0.65).astype(np.uint8)
own[prot_m] = (own[prot_m] * 0.35 + np.array([60, 220, 60]) * 0.65).astype(np.uint8)
own[hair_m & cloth_m] = (own[hair_m & cloth_m] * 0.15 + np.array([255, 255, 0]) * 0.85).astype(np.uint8)

panels = [
    ("ORIGINAL", np.asarray(orig_img)),
    ("RESULT (this run)", np.asarray(Image.open(RESULT).convert("RGB"))),
    ("ATR PARSE orange=hair blue=upper green=face", cls_rgb),
    ("MASKS r=hair b=clothing g=protected y=both", own),
]
scale = 2
cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
canvas = Image.new("RGB", (2 * cw + 18, 2 * (ch + 20) + 6), (18, 18, 22))
dr = ImageDraw.Draw(canvas)
for i, (title, arr) in enumerate(panels):
    tile = Image.fromarray(arr[y0:y1, x0:x1].astype(np.uint8)).resize(
        (cw, ch), Image.NEAREST
    )
    cx = (i % 2) * (cw + 6) + 6
    cy = (i // 2) * (ch + 20) + 6
    dr.text((cx + 4, cy + 4), title, fill=(255, 255, 255))
    canvas.paste(tile, (cx, cy + 16))
canvas.save(OUT / "ghost_attribution.png")
print(json.dumps(report, indent=1))
