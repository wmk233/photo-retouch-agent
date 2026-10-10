from __future__ import annotations
import numpy as np, cv2
from PIL import Image, ImageDraw

REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
blur = cv2.GaussianBlur(orig,(0,0),2.5)
sharp = cv2.addWeighted(orig,1.8,blur,-0.8,0)

X0,Y0,X1,Y1 = 400, 540, 620, 790
Z = 5
c = sharp[Y0:Y1, X0:X1]
im = Image.fromarray(c.astype(np.uint8)).resize(((X1-X0)*Z,(Y1-Y0)*Z), Image.NEAREST)
d = ImageDraw.Draw(im)
for x in range(X0, X1, 20):
    px = (x-X0)*Z
    d.line([(px,0),(px,im.height)], fill=(0,255,255), width=1)
    d.text((px+2,2), str(x), fill=(0,255,255))
for y in range(Y0, Y1, 20):
    py = (y-Y0)*Z
    d.line([(0,py),(im.width,py)], fill=(0,255,255), width=1)
    d.text((2,py+2), str(y), fill=(0,255,255))
im.save(f"{REPO}/evidence/p29-collar-adjudication/grid_orig_5x.png")
print("saved", im.size)
