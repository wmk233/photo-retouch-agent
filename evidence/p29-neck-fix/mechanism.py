"""Decisive mechanism probe: is the double hairline a *protected*-frozen fringe?

Offline only: one local SCHP inference is already cached as p29_labels.png, the
two real RunningHub results and the shipped masks are on disk. No provider call,
no spend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.edit_mask import build_semantic_edit_masks  # noqa: E402
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
RESULT = ROOT / "artifacts/p29-fix-run/result_0.png"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

orig_img = Image.open(ORIG).convert("RGB")
image = np.asarray(orig_img)
orig = image.astype(np.int16)
res = np.asarray(Image.open(RESULT).convert("RGB")).astype(np.int16)
orig_lum = np.asarray(orig_img.convert("L")).astype(np.int16)
res_lum = np.asarray(Image.open(RESULT).convert("L")).astype(np.int16)

labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
hair_class = labels == hp.HAIR
face_class = labels == hp.FACE
cloth_class = np.isin(labels, hp.CLOTHING_CLASSES)
bg_class = labels == hp.BACKGROUND

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)
mask_set = build_semantic_edit_masks(image, lm, labels)
hair_m = np.asarray(mask_set.hair) > 0
cloth_m = np.asarray(mask_set.clothing) > 0
prot_m = np.asarray(mask_set.protected) > 0
face_w = float(mask_set.face_width_px)
RIM = 0.12 * face_w

diff = np.abs(orig - res).max(axis=2)

hair_core = hair_class  # the parser's own hair silhouette
grown_hair = ndimage.binary_dilation(hair_core, iterations=int(round(RIM)))
fringe_in_guard = grown_hair & prot_m          # hair the guard FREEZES
frozen_parse_hair = hair_core & prot_m         # parsed hair the guard freezes

report: dict = {
    "face_width_px": round(face_w, 2),
    "hair_grow_rim_px": round(RIM, 1),
    "parse_hair_px": int(hair_class.sum()),
    "protected_px": int(prot_m.sum()),
    "protected_over_parse_hair_px": int(frozen_parse_hair.sum()),
    "protected_over_parse_hair_share_of_protected": round(
        float(frozen_parse_hair.sum()) / max(int(prot_m.sum()), 1), 4
    ),
    "grown_hair_frozen_by_protected_px": int(fringe_in_guard.sum()),
    "hair_mask_px": int(hair_m.sum()),
}

# how dark is the frozen fringe, and did the run change it?
if frozen_parse_hair.any():
    ys, xs = np.nonzero(frozen_parse_hair)
    report["protected_over_parse_hair_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    report["protected_over_parse_hair_mean_orig_lum"] = round(float(orig_lum[frozen_parse_hair].mean()), 1)
    report["protected_over_parse_hair_mean_res_lum"] = round(float(res_lum[frozen_parse_hair].mean()), 1)
    report["protected_over_parse_hair_identical_le2"] = round(
        float((diff[frozen_parse_hair] <= 2).mean()), 4
    )
    report["protected_over_parse_hair_mean_abs_delta"] = round(float(diff[frozen_parse_hair].mean()), 2)

# the dark-and-unchanged test on the forehead: parsed hair above the brow row
brow_row = int(round(float(lm[33][1]))) if lm.shape[0] > 33 else int(round(float(lm[:, 1].min())))
band = np.zeros((H, W), bool)
band[60:420, 200:720] = True
report["brow_row_landmark"] = brow_row
for name, sel in (
    ("band_frozen_parse_hair", band & frozen_parse_hair),
    ("band_hair_mask", band & hair_m),
    ("band_grown_hair_minus_protected", band & grown_hair & ~prot_m),
):
    n = int(sel.sum())
    if n == 0:
        report[name] = {"pixels": 0}
        continue
    report[name] = {
        "pixels": n,
        "mean_orig_lum": round(float(orig_lum[sel].mean()), 1),
        "mean_res_lum": round(float(res_lum[sel].mean()), 1),
        "unchanged_le2": round(float((diff[sel] <= 2).mean()), 4),
        "unchanged_le11": round(float((diff[sel] <= 11).mean()), 4),
    }

# where does the hair mask still sit on the garment / skin / background?
report["hair_mask_over_parse_clothing_px"] = int((hair_m & cloth_class).sum())
report["hair_mask_over_parse_face_px"] = int((hair_m & face_class).sum())
report["hair_mask_over_parse_face_share_of_hair"] = round(
    float((hair_m & face_class).sum()) / max(int(hair_m.sum()), 1), 4
)

(OUT / "mechanism.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))

# --------------------------------------------------------------------- figure
BOX = (200, 60, 720, 420)
SCALE = 3
ov = np.asarray(orig_img).copy()
ov[frozen_parse_hair] = (ov[frozen_parse_hair] * 0.30 + np.array([255, 0, 255]) * 0.70).astype(np.uint8)
ov[hair_m] = (ov[hair_m] * 0.55 + np.array([255, 60, 60]) * 0.45).astype(np.uint8)
g = np.zeros((H, W, 3), np.uint8)
g[frozen_parse_hair] = (255, 0, 255)
g[hair_m] = (255, 60, 60)
g[hair_m & frozen_parse_hair] = (255, 255, 0)
x0, y0, x1, y1 = BOX
cw, ch = (x1 - x0) * SCALE, (y1 - y0) * SCALE
panels = [("ORIGINAL", np.asarray(orig_img)), ("RESULT", np.asarray(Image.open(RESULT).convert("RGB"))),
          ("guard covers parsed hair (green=guard magenta=frozen hair)", ov),
          ("magenta=hair frozen by guard red=hair mask yellow=both", g)]
canvas = Image.new("RGB", (2 * cw + 18, 2 * (ch + 20) + 6), (18, 18, 22))
dr = ImageDraw.Draw(canvas)
for i, (title, arr) in enumerate(panels):
    tile = Image.fromarray(arr[y0:y1, x0:x1].astype(np.uint8)).resize((cw, ch), Image.NEAREST)
    cx, cy = (i % 2) * (cw + 6) + 6, (i // 2) * (ch + 20) + 6
    dr.text((cx + 4, cy + 4), title, fill=(255, 255, 255))
    canvas.paste(tile, (cx, cy + 16))
canvas.save(OUT / "mechanism_hairline.png")
print("saved", canvas.size)
