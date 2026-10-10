from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOX = (400, 540, 800, 990)  # x0,y0,x1,y1  chin / neck / chest / collar
SCALE = 3

o = Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB").crop(BOX)
r = Image.open(ROOT / "artifacts/p29-fix-run/result_0.png").convert("RGB").crop(BOX)
oa = np.asarray(o).astype(np.int16)
ra = np.asarray(r).astype(np.int16)
d = np.abs(oa - ra).max(axis=2)

hi = ra.astype(np.uint8).copy()
sel = d > 25
hi[sel] = (hi[sel].astype(np.float32) * 0.25 + np.array([255, 0, 0]) * 0.75).astype(np.uint8)
hi = np.clip(hi, 0, 255).astype(np.uint8)

w, h = o.size
w, h = w * SCALE, h * SCALE
panels = [("ORIGINAL 3x", o), ("THIS RUN 3x", r), ("RED = |delta|>25 3x", Image.fromarray(hi))]
canvas = Image.new("RGB", (2 * w + 18, 2 * (h + 20) + 6), (18, 18, 22))
dr = ImageDraw.Draw(canvas)
for i, (t, im) in enumerate(panels):
    cx = (i % 2) * (w + 6) + 6
    cy = (i // 2) * (h + 20) + 6
    dr.text((cx + 4, cy + 4), t, fill=(255, 255, 255))
    canvas.paste(im.resize((w, h), Image.NEAREST), (cx, cy + 16))
canvas.save(ROOT / "evidence/p29-neck-fix/neck_zoom3x.png")
print("saved", canvas.size)
