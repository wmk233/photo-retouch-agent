from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))

X0, Y0, X1, Y1 = 420, 560, 610, 760
Z = 7
def crop_up(arr):
    c = arr[Y0:Y1, X0:X1]
    im = Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width * Z, im.height * Z), Image.NEAREST)

po = crop_up(orig); pf = crop_up(final)
canvas = Image.new("RGB", (po.width * 2 + 10, po.height + 20), (30, 30, 30))
canvas.paste(po, (0, 20)); canvas.paste(pf, (po.width + 10, 20))
d = ImageDraw.Draw(canvas)
d.text((4, 4), "ORIGINAL 7x", fill=(255, 255, 255))
d.text((po.width + 14, 4), "FINAL 7x", fill=(255, 255, 255))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/tight_orig_final_7x.png")
print("saved", canvas.size, "crop", (X0, Y0, X1, Y1))
