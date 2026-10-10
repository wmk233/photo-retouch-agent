import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
X0,Y0,X1,Y1=498,562,586,632
Z=14
def panel(a,t):
    im=Image.fromarray(a[Y0:Y1,X0:X1]).resize(((X1-X0)*Z,(Y1-Y0)*Z),Image.NEAREST).convert("RGB")
    d=ImageDraw.Draw(im)
    for x in range(X0,X1+1,10):
        px=(x-X0)*Z; d.line([(px,0),(px,im.height)],fill=(0,255,255) if x%50==0 else (70,70,70)); d.text((px+2,2),str(x),fill=(255,255,0))
    for y in range(Y0,Y1+1,10):
        py=(y-Y0)*Z; d.line([(0,py),(im.width,py)],fill=(0,255,255) if y%50==0 else (70,70,70)); d.text((2,py+2),str(y),fill=(255,255,0))
    d.text((4,im.height-16),t,fill=(0,255,255))
    return im
a=panel(o,"ORIGINAL");b=panel(f,"FINAL")
cv=Image.new("RGB",(a.width*2+12,a.height+4),(15,15,15));cv.paste(a,(0,2));cv.paste(b,(a.width+12,2))
cv.save(REPO/"evidence/p29-collar-adjudication/chin_strip_14x.png");print("saved",cv.size)
