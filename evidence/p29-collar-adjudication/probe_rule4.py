from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
exec(open("probe_rule3.py").read().split("# overlay image")[0])
x0,y0,x1,y1=470,555,620,745
def panel_final(mask_rel, mask_keep, mask_held, title):
    im=f[y0:y1,x0:x1].copy()
    ov=im.copy()
    ov[mask_rel[y0:y1,x0:x1]]=(255,40,40)     # red = new release
    ov[mask_keep[y0:y1,x0:x1]]=(40,90,255)    # blue = kept guarded (real skin)
    ov[mask_held[y0:y1,x0:x1]]=(255,235,0)    # yellow = guarded not released & not skin-evidenced
    out=cv2.addWeighted(im,0.35,ov,0.65,0)
    p=out.copy(); cv2.putText(p,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(p,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return p
skinev = keep & (oC>=60)
other  = keep & ~(oC>=60)
S=6
panels=[
  cv2.resize(np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))[y0:y1,x0:x1],None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST),
  cv2.resize(f[y0:y1,x0:x1],None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST),
  cv2.resize(panel_final(release,skinev,other,"ADJUDICATION"),None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST),
  cv2.resize(panel_final(rel|release,np.zeros_like(keep),np.zeros_like(keep),"TOTAL EDIT REGION"),None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST),
]
big=np.concatenate(panels,axis=1)
Image.fromarray(big).save("v3_adjudication_6x.png")
print("wrote v3_adjudication_6x.png",big.shape,"crop",(x0,y0,x1,y1))
print("release",int(release.sum()),"skinev",int(skinev.sum()),"other",int(other.sum()))
