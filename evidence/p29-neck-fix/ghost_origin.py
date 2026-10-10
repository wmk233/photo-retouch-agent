"""Is the grey strip PRESERVED original pixels, or newly generated content?

preserved  -> the edit mask never covered it (protected leak / mask gap)
generated  -> a pass painted it and produced the sheen
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
o = np.asarray(Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.int16)
r = np.asarray(Image.open(ROOT / "artifacts/p29-fix-run/result_0.png").convert("RGB")).astype(np.int16)

raw_h = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127
raw_c = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127

def prep(m):
    return _mask_pipeline_np(m.astype(np.float32), blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                            expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                            fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"])) > 0.5
ph, pc = prep(raw_h), prep(raw_c)

d = np.abs(o - r).max(axis=2)

def region(name, x0, y0, x1, y1):
    sl = (slice(y0, y1), slice(x0, x1))
    sel = np.zeros(d.shape, bool); sel[sl] = True
    n = int(sel.sum())
    om = o[sl].reshape(-1, 3).mean(axis=0).round(1)
    rm = r[sl].reshape(-1, 3).mean(axis=0).round(1)
    return {
        "name": name, "box": [x0, y0, x1, y1], "pixels": n,
        "mean_abs_delta": round(float(d[sl].mean()), 2),
        "identical_frac": round(float((d[sl] <= 2).mean()), 4),
        "changed_gt25_frac": round(float((d[sl] > 25).mean()), 4),
        "orig_rgb": om.tolist(), "result_rgb": rm.tolist(),
        "prot_frac": round(float(prot[sl].mean()), 3),
        "hair_paint_frac": round(float(ph[sl].mean()), 3),
        "clothing_paint_frac": round(float(pc[sl].mean()), 3),
        "both_frac": round(float((ph[sl] & pc[sl]).mean()), 3),
        "neither_frac": round(float((~ph[sl] & ~pc[sl]).mean()), 3),
    }

out = {}
out["centre_ghost_column"] = region("centre grey column (chin->chest)", 500, 620, 660, 930)
out["chin_smear"] = region("under-chin smear", 340, 575, 620, 650)
out["left_chest_converted"] = region("left chest (should be olive)", 300, 780, 430, 930)
out["far_background"] = region("background right", 760, 700, 880, 900)

# The decisive split inside the ghost column: pixels the masks actually own.
sl = (slice(620, 930), slice(500, 660))
sel = np.zeros(d.shape, bool); sel[sl] = True
for label, mask in (("protected", prot & sel), ("hair_paint", ph & sel & ~prot),
                    ("clothing_paint_only", pc & sel & ~ph & ~prot), ("both_paint", ph & pc & sel & ~prot),
                    ("neither", sel & ~ph & ~pc & ~prot)):
    k = int(mask.sum())
    if not k:
        out[f"split::{label}"] = {"pixels": 0}; continue
    out[f"split::{label}"] = {
        "pixels": k,
        "mean_abs_delta": round(float(d[mask].mean()), 2),
        "identical_frac": round(float((d[mask] <= 2).mean()), 4),
        "orig_rgb": o[mask].reshape(-1, 3).mean(axis=0).round(1).tolist(),
        "result_rgb": r[mask].reshape(-1, 3).mean(axis=0).round(1).tolist(),
    }

# How much of the ORIGINAL off-white knit sitting under the protected mask survived?
knit = (o.min(axis=2) > 165) & (o.max(axis=2) > 200)
out["old_collar_total_px"] = int(knit.sum())
out["old_collar_inside_protected_px"] = int((knit & prot).sum())
out["old_collar_in_protected_still_white_frac"] = round(
    float((knit & prot & (d <= 25)).sum() / max(int((knit & prot).sum()), 1)), 4)

(ROOT / "evidence/p29-neck-fix/ghost_origin.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
