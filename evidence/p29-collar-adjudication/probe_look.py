import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
x0,y0,x1,y1=450,548,595,700
S=8
def grid(im,title,show_chin=True):
    p=cv2.resize(im[y0:y1,x0:x1],None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST).copy()
    for gx in range(x0,x1+1,10):
        c=(gx-x0)*S
        cv2.line(p,(c,0),(c,p.shape[0]),(0,0,0) if gx%50 else (0,255,0),1)
        if gx%50==0: cv2.putText(p,str(gx),(c+2,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0),1)
    for gy in range(y0,y1+1,10):
        c=(gy-y0)*S
        cv2.line(p,(0,c),(p.shape[1],c),(0,0,0) if gy%50 else (0,255,0),1)
        if gy%50==0: cv2.putText(p,str(gy),(2,c+12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0),1)
    if show_chin:
        cv2.circle(p,(int((486.61-x0)*S),int((594.19-y0)*S)),6,(255,0,0),-1)
    cv2.putText(p,title,(p.shape[1]-230,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return p
a=grid(o,"ORIGINAL x450-595 y548-700"); b=grid(f,"FINAL")
Image.fromarray(np.concatenate([a,b],axis=1)).save("v4_chin_10x.png")
print("wrote v4_chin_10x.png",a.shape,b.shape,"chin at",(486.61,594.19))
