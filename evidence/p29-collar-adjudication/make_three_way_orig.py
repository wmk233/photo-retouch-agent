from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
Z=8
def blk(img,y0,y1,x0,x1,label):
    c=img[y0:y1,x0:x1].copy()
    c=cv2.resize(c,((x1-x0)*Z,(y1-y0)*Z),interpolation=cv2.INTER_NEAREST)
    im=Image.fromarray(c); d=ImageDraw.Draw(im)
    d.rectangle([0,0,im.width-1,im.height-1],outline=(0,255,255),width=3)
    d.text((8,8),label,fill=(255,255,0))
    return im
panels=[
 blk(o,760,830,430,560,"ORIG knit sweater body"),
 blk(o,630,700,450,545,"ORIG old-release strip"),
 blk(o,594,726,470,566,"ORIG GUARDED band"),
 blk(o,548,592,455,530,"ORIG chin/jaw SKIN"),
 blk(f,594,726,470,566,"FINAL GUARDED band"),
 blk(f,548,592,455,530,"FINAL chin/jaw SKIN"),
]
H=max(p.height for p in panels); W=sum(p.width for p in panels)
out=Image.new("RGB",(W,H),(0,0,0)); x=0
for p in panels: out.paste(p,(x,0)); x+=p.width
out.save(REPO/"evidence/p29-collar-adjudication/three_way_orig_texture_8x.png")
print("saved",out.size)
