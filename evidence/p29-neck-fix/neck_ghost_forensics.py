"""Attribute the neck ghost to a specific mask (hair pass vs clothing pass).

Everything here is offline: original photo, the two real RunningHub results,
and the three accepted masks. No model re-runs, no spend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from tools.runninghub_crop_stitch_run import CROP_DEFAULTS, _mask_pipeline_np  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
RESULT = ROOT / "artifacts/p29-fix-run/result_0.png"
PREV = ROOT / "artifacts/p29-run/result_0.png"
OUT = ROOT / "evidence/p29-neck-fix"

orig = np.asarray(Image.open(ORIG).convert("RGB")).astype(np.int16)
res = np.asarray(Image.open(RESULT).convert("RGB")).astype(np.int16)
prev = np.asarray(Image.open(PREV).convert("RGB")).astype(np.int16)
lum_o = np.asarray(Image.open(ORIG).convert("L")).astype(np.int16)
lum_r = np.asarray(Image.open(RESULT).convert("L")).astype(np.int16)

H, W = lum_o.shape

raw_h = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127
raw_c = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127


def prep(mask: np.ndarray) -> np.ndarray:
    return _mask_pipeline_np(
        mask.astype(np.float32),
        blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
        expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
        fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"]),
    )


prep_h, prep_c = prep(raw_h), prep(raw_c)
paint_h = prep_h > 0.5
paint_c = prep_c > 0.5
soft_h = (prep_h > 0.0) & (prep_h <= 0.5)
soft_c = (prep_c > 0.0) & (prep_c <= 0.5)

diff = np.abs(orig - res).max(axis=2)
diff_prev = np.abs(orig - prev).max(axis=2)
lighter = (lum_r - lum_o).astype(np.int16)

# ---------------------------------------------------------------- components
try:
    from scipy import ndimage  # type: ignore

    def label(mask: np.ndarray):
        lab, n = ndimage.label(mask)
        return lab, n

    def component_stats(lab, n, score):
        rows = []
        for i in range(1, n + 1):
            sel = lab == i
            area = int(sel.sum())
            if area < 200:
                continue
            ys, xs = np.where(sel)
            rows.append(
                dict(
                    area=area,
                    bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    mean_lum_delta=float(lighter[sel].mean()),
                    mean_diff=float(diff[sel].mean()),
                )
            )
        rows.sort(key=lambda r: -r["area"])
        return rows
except Exception:  # pragma: no cover
    def label(mask):  # type: ignore
        return None, 0

    def component_stats(lab, n, score):  # type: ignore
        return []


def attribute(sel: np.ndarray) -> dict:
    total = int(sel.sum())
    if total == 0:
        return {"pixels": 0}
    both = int((sel & paint_h & paint_c).sum())
    only_h = int((sel & paint_h & ~paint_c).sum())
    only_c = int((sel & paint_c & ~paint_h).sum())
    neither = int((sel & ~paint_h & ~paint_c).sum())
    return dict(
        pixels=total,
        in_hair_paint=round(int((sel & paint_h).sum()) / total, 4),
        in_clothing_paint=round(int((sel & paint_c).sum()) / total, 4),
        both=round(both / total, 4),
        hair_only=round(only_h / total, 4),
        clothing_only=round(only_c / total, 4),
        neither=round(neither / total, 4),
        in_protected=round(int((sel & prot).sum()) / total, 4),
        in_soft_hair_ring=round(int((sel & soft_h).sum()) / total, 4),
        in_soft_clothing_ring=round(int((sel & soft_c).sum()) / total, 4),
    )


report: dict = {}

# The ghost reads as an added grey-white sheen below the chin.
ghost = (diff > 25) & (lighter > 15)
report["ghost_rule"] = "diff>25 and result is >15 luminance brighter than original"
report["ghost_pixels"] = int(ghost.sum())
lab, n = label(ghost)
rows = component_stats(lab, n, diff)
report["ghost_components"] = rows[:6]

if rows:
    big = rows[0]
    x0, y0, x1, y1 = big["bbox"]
    sel = np.zeros_like(ghost)
    sel[y0 : y1 + 1, x0 : x1 + 1] = ghost[y0 : y1 + 1, x0 : x1 + 1]
    report["primary_component_bbox"] = big["bbox"]
    report["primary_component"] = attribute(sel)

report["all_ghost"] = attribute(ghost)

# ------------------------------------------------- what the masks claim down there
band = np.zeros((H, W), bool)
band[560:860, 300:760] = True  # chin / neck / upper chest band
report["band_pixels"] = int(band.sum())
report["band_membership"] = attribute(band)

# the original white knit collar: near-white pixels in the band
near_white = (orig.min(axis=2) > 165) & (orig.max(axis=2) > 200)
collar = band & near_white
report["collar_pixels"] = int(collar.sum())
report["collar_membership"] = attribute(collar)

# rough skin proxy for the neck / chin skin we must NOT repaint
r, g, b = orig[..., 0], orig[..., 1], orig[..., 2]
skin = (
    (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b)
    & ((r.astype(np.int16) - g) > 15) & ((r.astype(np.int16) - b) > 15)
)
neck_skin = band & skin & ~prot
report["neck_skin_pixels"] = int(neck_skin.sum())
report["neck_skin_membership"] = attribute(neck_skin)

# central garment (below the collar, between the shoulders)
center = np.zeros((H, W), bool)
center[700:1000, 380:620] = True
report["center_garment_membership"] = attribute(center & ~skin)

(OUT / "neck_ghost_stats.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
