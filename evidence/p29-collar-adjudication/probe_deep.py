from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
which=sys.argv[1] if len(sys.argv)>1 else "orig"
src={"orig":REPO/"data/inputs/IMG_3651.jpg","final":REPO/"evidence/p29-submit-run/final.png"}[which]
a=np.asarray(Image.open(src).convert("RGB"))
X0,Y0,X1,Y1=495,585,590,705
Z=12
im=Image.fromarray(a[Y0:Y1,X0:X1]).resize(((X1-X0)*Z,(Y1-Y0)*Z),Image.NEAREST)
im=im.convert("RGB");d=ImageDraw.Draw(im)
for x in range(X0,X1+1,10):
    px=(x-X0)*Z;d.line([(px,0),(px,im.height)],fill=(0,255,255) if x%50==0 else (70,70,70),width=1)
    d.text((px+2,2),str(x),fill=(255,255,0))
for y in range(Y0,Y1+1,10):
    py=(y-Y0)*Z;d.line([(0,py),(im.width,py)],fill=(0,255,255) if y%50==0 else (70,70,70),width=1)
    d.text((2,py+2),str(y),fill=(255,255,0))
im.save(REPO/f"evidence/p29-collar-adjudication/deep_{which}_12x.png")
print("saved",im.size)
