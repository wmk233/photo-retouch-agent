from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
orig_p=REPO/"data/inputs/IMG_3651.jpg"
original=np.asarray(Image.open(orig_p).convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(orig_p),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; collar=np.asarray(ms.collar)>0; clothing=np.asarray(ms.clothing)>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
oL,oC=LC(original); fL,fC=LC(final)
print("collar px",int(collar.sum()),"clothing px",int(clothing.sum()))
print("protected bbox rows below chin:",int(protected[594:,:].sum()),"above 594:",int(protected[:594,:].sum()))
# what is protected around the strip?
sub=protected[570:600,515:565]
print("protected 570-600 x515-565 fraction",float(sub.mean()))
x0,y0,x1,y1=505,555,595,620; S=12
def mk(im):
    p=cv2.resize(im[y0:y1,x0:x1],None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST).copy()
    for gx in range(x0,x1+1,10):
        c=(gx-x0)*S; cv2.line(p,(c,0),(c,p.shape[0]),(0,90,0),1)
        cv2.putText(p,str(gx),(c+2,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    for gy in range(y0,y1+1,10):
        c=(gy-y0)*S; cv2.line(p,(0,c),(p.shape[1],c),(0,90,0),1)
        cv2.putText(p,str(gy),(2,c+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    cv2.line(p,(0,int((594.19-y0)*S)),(p.shape[1],int((594.19-y0)*S)),(255,0,0),1)  # chin row
    return p
fi=mk(final); ov=fi.copy()
cont,_=cv2.findContours((protected[y0:y1,x0:x1]*255).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ov,cont,-1,(255,255,0),1)
Image.fromarray(np.concatenate([mk(original),fi,ov],axis=1)).save(
    str(REPO/"evidence/p29-collar-adjudication/v5_guard_12x.png"))
print("wrote v5_guard_12x.png")
