from __future__ import annotations
import numpy as np, cv2, sys
from pathlib import Path
from PIL import Image
REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.face_lock import FACE_OVAL, face_width
from app.domains.portrait.edit_mask import fill_polygon, erode_mask, CHIN
from app.services.native_mask_inpainting import NativeMaskInpainting

orig = np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp  = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127

def lc(rgb):
    a=rgb.astype(np.float32); L=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]; C=a.max(2)-a.min(2); return L,C
def hue(rgb):
    a=rgb.astype(np.float32)/255.0; mx=a.max(2); mn=a.min(2); d=mx-mn
    r,g,b=a[...,0],a[...,1],a[...,2]; h=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; h[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; h[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; h[i]=60*((r[i]-g[i])/d[i])+240
    return h, np.where(m,d/np.maximum(mx,1e-6),0.0)
def tex(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap=cv2.Laplacian(g,cv2.CV_32F,ksize=3); return np.sqrt(cv2.GaussianBlur(lap*lap,(0,0),2.0))

ol,oc=lc(orig); fl,fc=lc(final); oh,os=hue(orig); ot=tex(orig)

svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(REPO/"data/inputs/IMG_3651.jpg",orig)
fw=face_width(lm)
oval=fill_polygon(orig.shape[:2], lm[list(FACE_OVAL)])
skin_face=erode_mask(oval,0.30*fw)>0

def region(x0,x1,y0,y1):
    m=np.zeros(orig.shape[:2],bool); m[y0:y1,x0:x1]=True; return m
P={
 "skin_face(oval)": skin_face,
 "skin_chin_jaw": region(448,486,556,588),
 "skin_cheek_L":  region(430,470,545,575),
 "knit_redband":  rel,
 "knit_cowl_R":   region(556,598,606,646),
 "knit_body_L":   region(424,466,700,752),
 "warm_center":   region(500,540,612,648),
 "guarded_all":   gp,
}
print(f"{'patch':16s} {'n':>6s} {'L':>6s} {'C':>6s} {'H':>6s} {'S':>6s} {'tex':>6s} {'|o-f|':>6s}")
for k,m in P.items():
    if m.sum()==0: print(k,"EMPTY"); continue
    dd=np.abs(orig.astype(np.float32)-final.astype(np.float32)).mean(2)[m].mean()
    print(f"{k:16s} {int(m.sum()):6d} {ol[m].mean():6.1f} {oc[m].mean():6.1f} {oh[m].mean():6.1f} {os[m].mean():6.2f} {ot[m].mean():6.1f} {dd:6.2f}")
