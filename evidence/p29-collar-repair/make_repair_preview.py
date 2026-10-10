"""Render the collar-repair preview: final, overlay, mask.

Purely a reporting aid: it loads the frozen final, the repair mask and the mask
stats, crops the collar neighbourhood and lays out three panels at 3x so the
operator can see (a) the residual as rendered, (b) what the single pass may
repaint (red) versus the guarded pale block it must leave (yellow), and (c) the
authored mask itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = REPO / "evidence/p29-submit-run/final.png"
MASK = HERE / "collar_repair_mask.png"
STATS = HERE / "collar_repair_stats.json"
OUT = HERE / "collar_repair_preview.png"

SCALE = 3
PAD = 12


def main() -> int:
    stats = json.loads(STATS.read_text())
    win = stats["window_bbox"]
    x0, y0 = max(0, win[0] - 60), max(0, win[1] - 70)
    x1, y1 = min(904, win[2] + 20), min(1190, win[3] + 40)

    final = Image.open(FINAL).convert("RGB").crop((x0, y0, x1, y1))
    mask = Image.open(MASK).convert("L").crop((x0, y0, x1, y1))
    guarded = Image.open(HERE / "collar_repair_guarded_pale.png").convert("L").crop((x0, y0, x1, y1))

    mask_b = np.array(mask) > 127
    guarded_b = np.array(guarded) > 127

    base = np.array(final).copy()
    overlay = base.copy()
    overlay[guarded_b] = (overlay[guarded_b] * 0.30 + np.array([255, 214, 0]) * 0.70).astype(np.uint8)
    overlay[mask_b] = (base[mask_b] * 0.30 + np.array([255, 40, 40]) * 0.70).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    w, h = final.size
    sw, sh = w * SCALE, h * SCALE
    panels = [
        ("final (frozen best render)", final.resize((sw, sh), Image.NEAREST)),
        ("release (red) vs guarded pale block (yellow)", overlay_img.resize((sw, sh), Image.NEAREST)),
        ("collar_repair_mask.png", Image.merge("RGB", [mask] * 3).resize((sw, sh), Image.NEAREST)),
    ]
    canvas = Image.new("RGB", (sw * 3 + PAD * 4, sh + 34 + PAD * 2), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(panels):
        px = PAD + i * (sw + PAD)
        canvas.paste(panel, (px, PAD + 24))
        draw.text((px, PAD + 6), title, fill=(235, 235, 235))
    canvas.save(OUT)
    print(f"wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
