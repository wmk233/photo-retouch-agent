from __future__ import annotations
import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
final=np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
gp=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png"))>127
def lc(a):
    return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
fl,cf=lc(final)
white=(cf<55)&(fl>150)
print("white px total", int(white.sum()))
print("white & guarded", int((white&gp).sum()), "of guarded", int(gp.sum()))
print("white & release", int((white&rel).sum()), "of release", int(rel.sum()))
print("guarded total", int(gp.sum()))
# how much of guarded is white-ish
for lim in (45,50,55,60,65,72):
    print(f"  guarded with final chroma<{lim}: {int((gp&(cf<lim)).sum())}")
# overlay
img=Image.fromarray(final.astype(np.uint8)).convert("RGB")
ov=np.asarray(img).copy()
ov[white]=(255,0,255)
def outline(mask,color,img2,th=2):
    u8=(mask*255).astype(np.uint8)
    cnts,_=cv2.findContours(u8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2,cnts,-1,color,th)
outline(rel,(255,0,0),ov); outline(gp,(0,200,255),ov)
X0,Y0,X1,Y1=420,575,600,760; Z=8
def cu(a):
    c=a[Y0:Y1,X0:X1]; im=Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width*Z,im.height*Z),Image.NEAREST)
p1=cu(np.asarray(img));p2=cu(ov)
W=p1.width;H=p1.height
canvas=Image.new("RGB",(W*2+10,H+22),(20,20,20));d=ImageDraw.Draw(canvas)
d.text((4,4),"FINAL",fill=(255,255,255))
d.text((W+14,4),"magenta=white(Ch<55,L>150) red=release blue=guarded",fill=(255,255,255))
canvas.paste(p1,(0,22));canvas.paste(p2,(W+10,22))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/white_mass_8x.png")
print("saved",canvas.size)
