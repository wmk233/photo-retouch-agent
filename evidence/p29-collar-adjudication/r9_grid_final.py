from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
S = 6
x0, y0, x1, y1 = 460, 555, 640, 735
def up(a):
    return Image.fromarray(a[y0:y1, x0:x1]).resize(((x1-x0)*S, (y1-y0)*S), Image.NEAREST)
panels = [up(orig), up(final)]
out = Image.new("RGB", (panels[0].width*2 + 10, panels[0].height), (0,0,0))
for i, p in enumerate(panels):
    d = ImageDraw.Draw(p)
    for gx in range(x0, x1, 20):
        X = (gx - x0) * S
        d.line([X, 0, X, p.height], fill=(0,255,255), width=1)
        d.text((X+2, 2), str(gx), fill=(0,255,255))
    for gy in range(y0, y1, 20):
        Y = (gy - y0) * S
        d.line([0, Y, p.width, Y], fill=(255,255,0), width=1)
        d.text((2, Y+2), str(gy), fill=(255,255,0))
    out.paste(p, (i*(p.width+10), 0))
out.save(REPO / "evidence/p29-collar-adjudication/r9_grid_orig_final_6x.png")
print("saved")
