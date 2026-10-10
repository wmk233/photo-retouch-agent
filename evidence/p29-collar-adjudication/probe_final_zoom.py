from __future__ import annotations
import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
final=np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
rel=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png"))>127
gp =np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
blur=cv2.GaussianBlur(final,(0,0),2.5); sharp=cv2.addWeighted(final,1.9,blur,-0.9,0)
def outline(mask,color,img,th=1):
    u8=(mask*255).astype(np.uint8)
    cnts,_=cv2.findContours(u8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,cnts,-1,color,th)
f1=final.copy(); outline(rel,(255,0,0),f1); outline(gp,(0,140,255),f1)
f2=sharp.copy(); outline(rel,(255,0,0),f2); outline(gp,(0,140,255),f2)
X0,Y0,X1,Y1=420,575,600,760
Z=8
def cu(a):
    c=a[Y0:Y1,X0:X1]; im=Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width*Z,im.height*Z),Image.NEAREST)
panels=[("FINAL 8x +outline",cu(f1)),("FINAL sharp 8x +outline",cu(f2))]
W=panels[0][1].width;H=panels[0][1].height
canvas=Image.new("RGB",(W*2+10,H+22),(20,20,20));d=ImageDraw.Draw(canvas)
for i,(t,im) in enumerate(panels):
    x=i*(W+10);canvas.paste(im,(x,22));d.text((x+4,4),t,fill=(255,255,255))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/final_zoom8_outline.png")
print("saved",canvas.size)
