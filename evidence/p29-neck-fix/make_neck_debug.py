"""One figure that shows the neck artifact next to the masks that own it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from tools.runninghub_crop_stitch_run import CROP_DEFAULTS, _mask_pipeline_np  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
OUT = ROOT / "evidence/p29-neck-fix"
ROI = (240, 420, 820, 920)  # x0, y0, x1, y1

orig = Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")
res = Image.open(ROOT / "artifacts/p29-fix-run/result_0.png").convert("RGB")
prev = Image.open(ROOT / "artifacts/p29-run/result_0.png").convert("RGB")

raw_h = np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127
raw_c = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127


def prep(m):
    return _mask_pipeline_np(m.astype(np.float32), blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                            expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                            fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"])) > 0.5


ph, pc = prep(raw_h), prep(raw_c)
crop = lambda im: im.crop(ROI)

panels = [("ORIGINAL", crop(orig)), ("PREV FAILED", crop(prev)), ("THIS RUN", crop(res))]

# mask overlay on this run
base = np.asarray(crop(res)).copy()
r = ROI[2] - ROI[0], ROI[3] - ROI[1]
owner = np.zeros((r[1], r[0], 3), np.uint8)
sh = (slice(ROI[1], ROI[3]), slice(ROI[0], ROI[2]))
owner[ph[sh]] = [255, 40, 40]        # hair paint  -> red
owner[pc[sh] & ~ph[sh]] = [40, 90, 255]  # clothing only -> blue
owner[prot[sh]] = [40, 230, 90]      # protected -> green
over = (base * 0.45 + owner * 0.55).astype(np.uint8)
panels.append(("MASK OWNER: red=hair blue=clothing-only", Image.fromarray(over)))

pad, label = 24, 18
cols, rows = 2, 2
w, h = r
canvas = Image.new("RGB", (cols * w + (cols + 1) * 6, rows * (h + label + pad) + 6), (20, 20, 24))
d = ImageDraw.Draw(canvas)
for i, (title, im) in enumerate(panels):
    cx = (i % cols) * (w + 6) + 6
    cy = (i // cols) * (h + label + pad) + 6
    d.text((cx + 4, cy + 4), title, fill=(255, 255, 255))
    canvas.paste(im, (cx, cy + label))
d.text((10, canvas.height - 14), "ROI x[240,820] y[420,920]", fill=(200, 200, 200))
canvas.save(OUT / "neck_debug.png")
print("wrote", OUT / "neck_debug.png", canvas.size)
