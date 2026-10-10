from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-collar-repair"
orig = Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")
final = Image.open(ROOT / "evidence/p29-submit-run/final.png").convert("RGB")
x0, y0, x1, y1 = 290, 540, 700, 850
scale = 2
panels = []
for name, im in [("ORIGINAL", orig), ("BEST FINAL", final)]:
    c = im.crop((x0, y0, x1, y1)).resize(((x1-x0)*scale, (y1-y0)*scale), Image.NEAREST)
    d = ImageDraw.Draw(c)
    d.text((8, 8), name, fill=(255, 0, 0))
    # grid every 50 source px
    for gx in range(x0, x1+1, 50):
        d.line([( (gx-x0)*scale, 0), ((gx-x0)*scale, (y1-y0)*scale)], fill=(0,255,0), width=1)
        d.text(((gx-x0)*scale+2, 2), str(gx), fill=(0,255,0))
    for gy in range(y0, y1+1, 50):
        d.line([(0, (gy-y0)*scale), ((x1-x0)*scale, (gy-y0)*scale)], fill=(0,255,0), width=1)
        d.text((2, (gy-y0)*scale+2), str(gy), fill=(0,255,0))
    panels.append(c)
w = panels[0].width; h = panels[0].height
canvas = Image.new("RGB", (w*2+10, h), (30,30,30))
canvas.paste(panels[0], (0,0)); canvas.paste(panels[1], (w+10,0))
canvas.save(OUT / "neck_zoom_orig_vs_final_2x.png")
print("saved", canvas.size)
