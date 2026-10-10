from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
orig=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127

X0,Y0,X1,Y1=425,570,605,765
Z=7
def crop(a): 
    return Image.fromarray(a[Y0:Y1,X0:X1])
def up(im): return im.resize((im.width*Z,im.height*Z),Image.NEAREST)
def gridded(a,title):
    im=crop(a); im=im.resize((im.width*Z,im.height*Z),Image.NEAREST).convert("RGB")
    d=ImageDraw.Draw(im)
    for x in range(X0,X1+1,10):
        col=(255,255,255) if (x%50==0) else (90,90,90)
        px=(x-X0)*Z; d.line([(px,0),(px,im.height)],fill=col,width=1)
        if x%50==0: d.text((px+2,2),str(x),fill=(255,255,0))
    for y in range(Y0,Y1+1,10):
        col=(255,255,255) if (y%50==0) else (90,90,90)
        py=(y-Y0)*Z; d.line([(0,py),(im.width,py)],fill=col,width=1)
        if y%50==0: d.text((2,py+2),str(y),fill=(255,255,0))
    d.text((4,im.height-16),title,fill=(0,255,255))
    return im
p_o=gridded(orig,"ORIGINAL")
p_f=gridded(final,"FINAL (frozen)")
# marker overlay on final: guarded=orange outline, release=magenta outline
import cv2
ov=final.copy()
def outline(mask,color,th=1):
    u8=(mask*255).astype(np.uint8)
    cnts,_=cv2.findContours(u8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnts,-1,color,th)
outline(gp,(255,140,0)); outline(rel,(255,0,255))
# fill translucent
ovf=ov.astype(np.float32); m=gp[...,None]; mfill=(np.array([255,120,0],np.float32))[None,None,:]
ovf=np.where(m, 0.55*final+0.45*np.broadcast_to(mfill,final.shape), ovf).astype(np.uint8)
p_g=gridded(ovf,"FINAL + guarded(orange fill) release(magenta line)")
W=p_o.width;H=p_o.height
cv=Image.new("RGB",(W*3+20,H+20),(15,15,15))
cv.paste(p_o,(0,20));cv.paste(p_f,(W+10,20));cv.paste(p_g,(2*W+20,20))
cv.save(REPO/"evidence/p29-collar-adjudication/sbs_orig_final_guarded_7x.png")
print("saved",cv.size, "guarded px",int(gp.sum()),"release px",int(rel.sum()))
