from __future__ import annotations
import numpy as np, cv2, sys
from pathlib import Path
from PIL import Image, ImageDraw
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.face_lock import FACE_OVAL, face_width
from app.domains.portrait.edit_mask import fill_polygon, erode_mask
from app.services.native_mask_inpainting import NativeMaskInpainting

orig_u8=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
orig=orig_u8.astype(np.float32)
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127

def lcs(a):
    L=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]; C=a.max(2)-a.min(2)
    mx=a.max(2); S=np.where(mx>1e-6,C/np.maximum(mx,1e-6),0.0)
    return L,C,S
oL,oC,oS=lcs(orig); fL,fC,fS=lcs(final)

svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(REPO/"data/inputs/IMG_3651.jpg",orig_u8)
fw=face_width(lm); oval=fill_polygon(orig.shape[:2],lm[list(FACE_OVAL)])
skin_face=erode_mask(oval,0.30*fw)>0
def region(x0,x1,y0,y1):
    m=np.zeros(orig.shape[:2],bool); m[y0:y1,x0:x1]=True; return m
knit_ref = rel | region(420,470,700,755)
skin_ref = skin_face | region(448,486,556,588)

def pct(m,name):
    print(f"{name:14s} n={int(m.sum()):5d}  C p10/50/90 = {np.percentile(oC[m],10):5.1f}/{np.percentile(oC[m],50):5.1f}/{np.percentile(oC[m],90):5.1f}"
          f"   S p10/50/90 = {np.percentile(oS[m],10):5.2f}/{np.percentile(oS[m],50):5.2f}/{np.percentile(oS[m],90):5.2f}"
          f"   L p50={np.percentile(oL[m],50):6.1f}")
pct(skin_ref,"skin_ref")
pct(knit_ref,"knit_ref")
pct(gp,"guarded")

# classify guarded by nearest reference in (C/255, S) normalized space
mu_s=np.array([np.median(oC[skin_ref]), np.median(oS[skin_ref])])
mu_k=np.array([np.median(oC[knit_ref]), np.median(oS[knit_ref])])
sd=np.array([np.std(oC[skin_ref]), np.std(oS[skin_ref])])
kd=np.array([np.std(oC[knit_ref]), np.std(oS[knit_ref])])
F=np.stack([oC,oS],2)
d_s=np.linalg.norm((F-mu_s)/np.maximum(sd,1e-3),axis=2)
d_k=np.linalg.norm((F-mu_k)/np.maximum(kd,1e-3),axis=2)
print("skin mu",mu_s.round(1),"sd",sd.round(1))
print("knit mu",mu_k.round(1),"sd",kd.round(1))

# release = cloth-evidence: closer to knit by a margin AND final pale (C<60) AND final bright(L>160)
cloth = (d_k < d_s - 0.35) & (fC < 60) & (fL > 160) & gp
skin  = (d_s < d_k - 0.5) & gp
unc   = gp & ~cloth & ~skin
print("guarded split: cloth",int(cloth.sum())," skin",int(skin.sum())," uncertain",int(unc.sum()))

def outline(mask,color,img,th=2):
    u8=(mask*255).astype(np.uint8)
    cnts,_=cv2.findContours(u8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,cnts,-1,color,th)
img=final.astype(np.uint8).copy()
img[cloth]=(255,0,0); img[skin]=(0,255,0)
ov2=final.astype(np.uint8).copy()
outline(gp,(0,200,255),ov2,2); outline(rel,(255,0,255),ov2,2)
ov2[cloth]=(255,0,0); ov2[skin]=(0,255,0)
X0,Y0,X1,Y1=420,575,600,760;Z=8
def cu(a):
    c=a[Y0:Y1,X0:X1];im=Image.fromarray(c.astype(np.uint8));return im.resize((im.width*Z,im.height*Z),Image.NEAREST)
p1=cu(img);p2=cu(ov2)
W=p1.width;H=p1.height
cv=Image.new("RGB",(W*2+10,H+22),(20,20,20));d=ImageDraw.Draw(cv)
d.text((4,4),"RED=cloth-evidence release  GREEN=skin keep",fill=(255,255,255))
d.text((W+14,4),"same + blue=guarded outline magenta=release outline",fill=(255,255,255))
cv.paste(p1,(0,22));cv.paste(p2,(W+10,22))
cv.save(REPO/"evidence/p29-collar-adjudication/classify_8x.png")
print("saved",cv.size)
