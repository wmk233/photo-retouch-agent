"""Hairline forensics: the forehead band, 3x, with the mask boundary drawn."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.domains.portrait import human_parsing as hp  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
RESULT = ROOT / "artifacts/p29-fix-run/result_0.png"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

BOX = (200, 60, 720, 420)  # x0,y0,x1,y1  hairline / forehead band
SCALE = 3

orig_img = Image.open(ORIG).convert("RGB")
res_img = Image.open(RESULT).convert("RGB")
orig = np.asarray(orig_img)
res = np.asarray(res_img)
labels = np.asarray(Image.open(LABELS))
hair_class = labels == hp.HAIR
face_class = labels == hp.FACE
upper_class = labels == hp.UPPER_CLOTHES

hair_m = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127
prot_m = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127

# 1 px boundary of each region, so the seam is visible at 3x
from scipy import ndimage  # noqa: E402

def edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=1)

overlay = orig.copy()
overlay[hair_m] = (overlay[hair_m] * 0.45 + np.array([255, 40, 40]) * 0.55).astype(np.uint8)
overlay[prot_m] = (overlay[prot_m] * 0.45 + np.array([40, 220, 40]) * 0.55).astype(np.uint8)
overlay[edge(hair_m)] = np.array([255, 255, 0], np.uint8)
overlay[edge(prot_m)] = np.array([0, 255, 255], np.uint8)

parse_rgb = np.zeros((*labels.shape, 3), np.uint8)
parse_rgb[hair_class] = (255, 150, 40)
parse_rgb[face_class] = (60, 220, 90)
parse_rgb[upper_class] = (70, 130, 255)

diff = np.abs(orig.astype(np.int16) - res.astype(np.int16)).max(axis=2)
dimg = np.asarray(orig).astype(np.uint8).copy()
dimg = (dimg * 0.35).astype(np.uint8)
dimg[diff > 25] = np.array([255, 0, 0], np.uint8)

x0, y0, x1, y1 = BOX
cw, ch = (x1 - x0) * SCALE, (y1 - y0) * SCALE
panels = [
    ("ORIGINAL 3x", orig),
    ("RESULT this run 3x", res),
    ("HAIR MASK (+red) PROTECTED (green) edges y/c", overlay),
    ("|delta|>25 red 3x", dimg),
    ("ATR PARSE orange=hair green=face blue=upper", parse_rgb),
]
canvas = Image.new("RGB", (2 * cw + 18, 3 * (ch + 20) + 6), (18, 18, 22))
dr = ImageDraw.Draw(canvas)
for i, (title, arr) in enumerate(panels):
    tile = Image.fromarray(np.asarray(arr)[y0:y1, x0:x1].astype(np.uint8)).resize(
        (cw, ch), Image.NEAREST
    )
    cx = (i % 2) * (cw + 6) + 6
    cy = (i // 2) * (ch + 20) + 6
    dr.text((cx + 4, cy + 4), title, fill=(255, 255, 255))
    canvas.paste(tile, (cx, cy + 16))
canvas.save(OUT / "hairline_zoom3x.png")

# ------------------------------------------------------------- numeric facts
band = np.zeros(labels.shape, bool)
band[y0:y1, x0:x1] = True
hair_rows = np.where(hair_m.any(axis=1))[0]
facts = {
    "box": list(BOX),
    "hair_mask_top_row": int(hair_rows.min()),
    "protected_top_row": int(np.where(prot_m.any(axis=1))[0].min()),
    "parse_hair_top_row": int(np.where(hair_class.any(axis=1))[0].min()),
    "band_pixels": int(band.sum()),
    "band_hair_mask": int((band & hair_m).sum()),
    "band_protected": int((band & prot_m).sum()),
    "band_parse_hair": int((band & hair_class).sum()),
    "band_parse_face": int((band & face_class).sum()),
    "band_changed_gt25": int((band & (diff > 25)).sum()),
    "band_changed_gt25_in_hair_mask": int((band & (diff > 25) & hair_m).sum()),
    "band_changed_gt25_outside_hair_mask": int((band & (diff > 25) & ~hair_m).sum()),
    "hair_mask_over_parse_face_px": int((hair_m & face_class).sum()),
    "changed_gt25_in_protected_px": int((diff > 25) & prot_m).sum() if False else int(
        ((diff > 25) & prot_m).sum()
    ),
    "protected_px": int(prot_m.sum()),
    "protected_identical_le2": round(float((diff[prot_m] <= 2).mean()), 4),
    "protected_identical_le11": round(float((diff[prot_m] <= 11).mean()), 4),
    "protected_max_delta": int(diff[prot_m].max()),
}
(OUT / "hairline_stats.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1))
