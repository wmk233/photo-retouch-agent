from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
oL,oC=0,0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
def HS(a):
    a=a.astype(np.float32)/255.0; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.0)
    r,g,b=a[...,0],a[...,1],a[...,2]; h=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; h[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; h[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; h[i]=60*((r[i]-g[i])/d[i])+240
    return h, np.where(m,sat,0.0)
def lapvar(img,y0,y1,x0,x1):
    g=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g,cv2.CV_32F).var())
def med(img,y0,y1,x0,x1):
    L,C=LC(img[y0:y1,x0:x1]); H,S=HS(img[y0:y1,x0:x1])
    return (round(float(np.median(L)),1),round(float(np.median(C)),1),round(float(np.median(H)),1),round(float(np.median(S)),3))
regs={
 "cheek/face skin      ": (505,560,470,540),
 "jaw skin (above chin)": (555,585,455,520),
 "chin skin            ": (570,592,470,540),
 "sweater body (below) ": (760,830,430,560),
 "sweater chest left   ": (700,760,400,470),
 "OLD release strip    ": (635,705,450,535),
 "guarded block center ": (605,690,525,562),
 "guarded block left   ": (605,665,440,520),
 "under-chin warm band ": (592,615,470,525),
}
print("%-22s %-28s %-28s %s"%("region","ORIG med(L,C,H,S)","FINAL med(L,C,H,S)","lapvar orig/final"))
for name,(y0,y1,x0,x1) in regs.items():
    print("%-22s %-28s %-28s %.1f / %.1f"%(name,med(o,y0,y1,x0,x1),med(f,y0,y1,x0,x1),lapvar(o,y0,y1,x0,x1),lapvar(f,y0,y1,x0,x1)))
