from __future__ import annotations
import sys, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, FACE_OVAL, build_semantic_edit_masks
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour, fill_polygon, fill_feature_guard, dilate_mask)
from app.services.native_mask_inpainting import NativeMaskInpainting
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
collar=np.asarray(ms.collar)>0; clothing=np.asarray(ms.clothing)>0
print("collar px",int(collar.sum()),"clothing px",int(clothing.sum()),"protected px",int(protected.sum()))
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
oval=fill_polygon((h,w),lm[list(FACE_OVAL)])>0
features=dilate_mask(fill_feature_guard((h,w),lm), 0.02*fw)>0
ref,_=skin_reference_colour(original,lm,fw); skin=adaptive_skin_mask(original,ref,fw,config_from_settings(settings))>0
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
def HS(a):
    a=a.astype(np.float32)/255.; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.)
    r,g,b=a[...,0],a[...,1],a[...,2]; hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue,np.where(m,sat,0.)
oL,oC=LC(original); fL,fC=LC(final); fH,fS=HS(final)
refL,refC=LC(np.asarray(ref,dtype=np.float32).reshape(1,1,3)); refC=float(refC[0,0])
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72)
guarded=pale&~background&~hair&~olive&skin&win
diff=np.abs(original.astype(np.int16)-final.astype(np.int16)).max(2); unch=diff<10

# texture: local std at 3x3 of gray, and gradient magnitude
def tex(img):
    g=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    m=cv2.blur(g,(3,3)); m2=cv2.blur(g*g,(3,3)); return np.sqrt(np.maximum(m2-m*m,0))
tO=tex(original); tF=tex(final)
def patch(x0,x1,y0,y1):
    m=np.zeros((h,w),bool); m[y0:y1,x0:x1]=True; return m
sweater=patch(120,300,700,820)&~background
guarded_core=patch(510,570,600,695)&guarded
face_skin=patch(int(cx-60),int(cx+60),int(cy-150),int(cy-90))&oval&skin
neck_skin_final=patch(492,516,620,660)&skin&~pale
print("tex(3x3 std) medians  orig: sweater",round(float(np.median(tO[sweater])),2),
      " guarded",round(float(np.median(tO[guarded_core])),2),
      " face_skin",round(float(np.median(tO[face_skin])),2))
print("tex(3x3 std) medians  final: guarded",round(float(np.median(tF[guarded_core])),2),
      " neck_skin",round(float(np.median(tF[neck_skin_final])),2),
      " sweater_orig_area_final",round(float(np.median(tF[sweater])),2))
for nm,m in (("sweater",sweater),("guarded_core",guarded_core),("face_skin",face_skin)):
    if m.any(): print(f"  {nm:12s} n={int(m.sum())} texO p10/50/90=",[round(float(np.percentile(tO[m],q)),2) for q in (10,50,90)])

# neck skin chroma floor in FINAL across the collar band
band=win&skin&~pale&~hair&~background
print("FINAL collar-band non-pale skin px",int(band.sum()),"chroma p1/p5/p10/50=",[round(float(np.percentile(fC[band],q))) for q in (1,5,10,50)])
# candidate rule
ratio=0.80
cand=guarded&unch&(fC<=ratio*refC)&(oC<=ratio*refC)&~oval&~features&~collar&~clothing
print("CAND px",int(cand.sum()),"of guarded",int(guarded.sum()))
keep=guarded&~cand
print("KEEP px",int(keep.sum()))
n,lab,st,_=cv2.connectedComponentsWithStats(cand.astype(np.uint8),8)
rows=sorted([{"px":int(st[i,4]),"bbox":[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]} for i in range(1,n)],key=lambda r:-r["px"])
print("CAND blobs:",json.dumps(rows[:10]))
n,lab,st,_=cv2.connectedComponentsWithStats(keep.astype(np.uint8),8)
rows=sorted([{"px":int(st[i,4]),"bbox":[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])],
   "FIN":[int(v) for v in np.round(final[lab==i].mean(0))]} for i in range(1,n)],key=lambda r:-r["px"])
print("KEEP blobs:",json.dumps(rows[:10]))
