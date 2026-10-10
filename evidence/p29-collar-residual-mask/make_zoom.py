"""Clean high-quality zooms of the collar strip from the original and the final."""
import numpy as np
from PIL import Image, ImageDraw

REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin = np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
print("orig", orig.shape, "final", fin.shape)

def crop(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1]

def up(arr, k):
    im = Image.fromarray(arr)
    w, h = im.size
    return im.resize((w * k, h * k), Image.LANCZOS)

def band(panels, labels, gap=14, bg=(24, 24, 24)):
    hs = [p.size[1] + 30 for p in panels]
    ws = [p.size[0] for p in panels]
    H = max(hs); W = sum(ws) + gap * (len(panels) - 1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for p, lb in zip(panels, labels):
        out.paste(p, (x, 30))
        ImageDraw.Draw(out).text((x + 6, 9), lb, fill=(255, 240, 120))
        x += p.size[0] + gap
    return out

for name, (x0, y0, x1, y1), k in [
    ("wide", (380, 520, 700, 800), 4),
    ("tight", (490, 560, 630, 730), 8),
]:
    o = up(crop(orig, x0, y0, x1, y1), k)
    f = up(crop(fin, x0, y0, x1, y1), k)
    band([o, f], [f"ORIGINAL {k}x {x0},{y0}-{x1},{y1}", f"FINAL {k}x {x0},{y0}-{x1},{y1}"]).save(
        f"{REPO}/evidence/p29-collar-residual-mask/zoom_{name}_{k}x.png")
    print("wrote", name)
