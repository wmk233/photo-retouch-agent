"""Free-gate evidence: the collar repair mask before vs after this round's carve.

Three panels at 8x, cropped to the collar neighbourhood:
  1. the frozen best render (final.png),
  2. the mask as it stood before this round (release without the carve) in red,
  3. the mask after this round: the previous release in red *plus* the newly
     carved protected strip in green.

"Before" is reconstructed exactly as ``new mask minus the carve`` -- the two share
every other pixel, so the green in panel 3 is precisely what this round added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
MASK = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_mask.png").convert("L")) > 0
CARVE = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_carve.png").convert("L")) > 0

X0, Y0, X1, Y1 = 470, 556, 612, 700
SCALE = 8
PAD = 14


def blend(base: np.ndarray, mask: np.ndarray, colour, alpha: float) -> np.ndarray:
    out = base.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.asarray(colour) * alpha).astype(np.uint8)
    return out


before = MASK & ~CARVE
final_c = FINAL.copy()
p1 = final_c
p2 = blend(final_c, before, (255, 40, 40), 0.6)
p3 = blend(blend(final_c, before, (255, 40, 40), 0.6), CARVE, (40, 255, 60), 0.85)

def crop8(img):
    c = img[Y0:Y1, X0:X1]
    return Image.fromarray(c).resize((c.shape[1] * SCALE, c.shape[0] * SCALE), Image.NEAREST)

tiles = [
    ("final (frozen best render)", crop8(p1)),
    ("mask BEFORE this round (release, red)", crop8(p2)),
    ("mask AFTER (release red + new carve green)", crop8(p3)),
]
w, h = tiles[0][1].size
canvas = Image.new("RGB", (w * 3 + PAD * 4, h + 40 + PAD * 2), (22, 22, 26))
draw = ImageDraw.Draw(canvas)
for i, (title, tile) in enumerate(tiles):
    x = PAD + i * (w + PAD)
    canvas.paste(tile, (x, PAD + 26))
    draw.text((x, PAD + 8), title, fill=(238, 238, 238))
out = HERE / "carve_before_after_8x.png"
canvas.save(out)
print("wrote", out, canvas.size)
print("release px", int(MASK.sum()), "| carve px", int(CARVE.sum()), "| before px", int(before.sum()))
