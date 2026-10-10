"""Free-gate evidence: is any visible white cloth left in the collar band?

Two panels at 8x over the same crop:
  1. the frozen best final with the adjudication classes painted on it --
     red    = collar_repair_mask (released, this pass repaints it)
     green  = this round's carve (protected pixels moved into the release)
     blue   = guarded_pale (skin-toned, kept)
     magenta= strictly neutral white (final luma > 160 and chroma < 20) -- the
              pixels that would read as white cloth. There are ten in the band,
              each <= 4 px, and they sit on the pale-olive blouse edge.
  2. the same crop of the ORIGINAL photo, so the operator can see that the warm
     tone under the jaw is the subject's neck/skin, not the white sweater.

Read with ``collar_repair_stats.json``: neutral_white_uncovered_px, and
gate_report.json: gate8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MASK = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_mask.png").convert("L")) > 127
CARVE = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_carve.png").convert("L")) > 127
GUARD = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_guarded_pale.png").convert("L")) > 127
FINAL = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
ORIG = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))

f = FINAL.astype(np.float32)
luma = f.mean(2)
chroma = f.max(2) - f.min(2)
NEUTRAL = (luma > 160) & (chroma < 20)

X0, Y0, X1, Y1 = 430, 560, 660, 780
SCALE = 8
PAD = 14


def blend(base, mask, colour, alpha):
    out = base.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.asarray(colour) * alpha).astype(np.uint8)
    return out


painted = blend(FINAL, MASK, (255, 40, 40), 0.55)
painted = blend(painted, CARVE, (40, 255, 60), 0.85)
painted = blend(painted, GUARD, (60, 120, 255), 0.45)
painted = blend(painted, NEUTRAL, (255, 0, 255), 0.95)


def crop8(img):
    c = img[Y0:Y1, X0:X1]
    return Image.fromarray(c).resize((c.shape[1] * SCALE, c.shape[0] * SCALE), Image.NEAREST)


tiles = [
    ("final + adjudication: red=release green=carve blue=guarded magenta=neutral-white", crop8(painted)),
    ("ORIGINAL photo, same crop", crop8(ORIG)),
]
w, h = tiles[0][1].size
canvas = Image.new("RGB", (w * 2 + PAD * 3, h + 40 + PAD * 2), (22, 22, 26))
draw = ImageDraw.Draw(canvas)
for i, (title, tile) in enumerate(tiles):
    x = PAD + i * (w + PAD)
    canvas.paste(tile, (x, PAD + 26))
    draw.text((x, PAD + 8), title, fill=(238, 238, 238))
out = HERE / "residual_white_evidence_8x.png"
canvas.save(out)
sub = np.zeros_like(NEUTRAL)
sub[Y0:Y1, X0:X1] = True
print("wrote", out, canvas.size)
print("mask", int(MASK.sum()), "carve", int(CARVE.sum()), "guard", int(GUARD.sum()),
      "neutral_white in crop", int((NEUTRAL & sub).sum()))
