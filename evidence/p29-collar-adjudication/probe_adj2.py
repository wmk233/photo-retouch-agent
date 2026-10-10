from __future__ import annotations
import sys, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour, _ycc)
from app.services.native_mask_inpainting import NativeMaskInpainting

original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
clothing=np.asarray(ms.clothing)>0; collar=np.asarray(ms.collar)>0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
ref,nref=skin_reference_colour(original,lm,fw); 
skin=adaptive_skin_mask(original,ref,fw,config_from_settings(settings))>0
print("face_width",round(fw,2),"chin",round(cx,2),round(cy,2),"skin_ref rgb",np.round(ref,1),"n",nref)
rL,rCr,rCb=_ycc(np.asarray(ref,dtype=np.float32).reshape(1,1,3))
print("skin_ref YCrCb",round(float(rL[0,0]),1),round(float(rCr[0,0]),1),round(float(rCb[0,0]),1))

def LC(a):
    a=a.astype(np.float32)
    return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
def HS(a):
    a=a.astype(np.float32)/255.; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.)
    r,g,b=a[...,0],a[...,1],a[...,2]; hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue,np.where(m,sat,0.)
oL,oC=LC(original); fL,fC=LC(final); fH,fS=HS(final)
yL,yCr,yCb=_ycc(original)
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72); sknit=(oL>140)&(oC<50)
guarded=pale&~background&~hair&~olive&skin&win
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
diff=np.abs(original.astype(np.int16)-final.astype(np.int16)).max(2)
unchanged=diff<10

# texture energy: high-pass on the ORIGINAL (no generator smoothing) and on FINAL
def hf(img):
    g=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.abs(g-cv2.medianBlur(g.astype(np.uint8),5).astype(np.float32))
hfO=hf(original); hfF=hf(final)

def rep(m,name):
    if not m.any(): print(f"{name:16s} EMPTY"); return
    ys,xs=np.where(m)
    print(f"{name:16s} px={int(m.sum()):5d} bbox=[{xs.min()},{ys.min()},{xs.max()},{ys.max()}]")
    print(f"{'':16s}  ORIG {np.round(original[m].mean(0)).astype(int)} oL{round(float(np.median(oL[m])))} oC{round(float(np.median(oC[m])))}"
          f" | FIN {np.round(final[m].mean(0)).astype(int)} fL{round(float(np.median(fL[m])))} fC{round(float(np.median(fC[m])))} fH{round(float(np.median(fH[m])))}")
    print(f"{'':16s}  unchanged_frac={float(unchanged[m].mean()):.3f}  hfOrig_med={float(np.median(hfO[m])):.2f} hfFinal_med={float(np.median(hfF[m])):.2f}")

# references
chin_skin=np.zeros((h,w),bool); chin_skin[int(cy-34):int(cy-6),int(cx-40):int(cx+40)]=True
chin_skin&=skin&pale
olive_m=olive&(np.arange(h)[:,None]>cy)&(np.arange(h)[:,None]<cy+0.9*fw)
rep(guarded,"guarded")
rep(rel&pale,"release_pale")
rep(chin_skin,"chin_skin")
rep(olive_m,"olive_blouse")

# SCHP label histogram inside guarded
lab_names={0:"background",1:"hat",2:"hair",3:"sunglasses",4:"upper-clothes",5:"skirt",6:"pants",7:"dress",8:"belt",9:"left-shoe",10:"right-shoe",11:"face",12:"left-leg",13:"right-leg",14:"left-arm",15:"right-arm",16:"bag",17:"scarf"}
gl=labels[guarded]
u,c=np.unique(gl,return_counts=True)
print("guarded SCHP labels:", {int(k):int(v) for k,v in zip(u,c)})
print("  ->", {lab_names.get(int(k),str(k)):int(v) for k,v in zip(u,c)})
print("guarded protected frac", float(protected[guarded].mean()), "collar", float(collar[guarded].mean()), "clothing", float(clothing[guarded].mean()))
print("guarded hair",int((guarded&hair).sum()),"bg",int((guarded&background).sum()))

# whitish-looking fabric inside window in FINAL (visible pale cloth)
white_final=(fL>168)&(fC<45)&win&~background&~hair&~olive
print("FINAL whitish in window px",int(white_final.sum()),
      "guarded",int((white_final&guarded).sum()),
      "release",int((white_final&rel).sum()),
      "other",int((white_final&~guarded&~rel).sum()))
# blobs of whitish guarded
m=(white_final&guarded).astype(np.uint8)
n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
bl=sorted([{"px":int(st[i,4]),"bbox":[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]} for i in range(1,n)],key=lambda b:-b["px"])
print("whitish&guarded blobs:",json.dumps(bl[:12]))

# per-blob of guarded
n,lab,st,_=cv2.connectedComponentsWithStats(guarded.astype(np.uint8),8)
rows=[]
for i in range(1,n):
    mm=lab==i; px=int(st[i,4])
    if px<60: continue
    rows.append({"px":px,"bbox":[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])],
        "ORIG":[int(v) for v in np.round(original[mm].mean(0))],
        "FIN":[int(v) for v in np.round(final[mm].mean(0))],
        "oC":round(float(np.median(oC[mm]))),"fC":round(float(np.median(fC[mm]))),
        "fH":round(float(np.median(fH[mm]))),
        "unch":round(float(unchanged[mm].mean()),2),
        "hfO":round(float(np.median(hfO[mm])),1),
        "schp":int(np.bincount(labels[mm].ravel()).argmax()),
        "skinflag":round(float(skin[mm].mean()),2)})
rows.sort(key=lambda r:-r["px"])
print("guarded blobs >=60px:")
for r in rows[:14]: print("  ",json.dumps(r))
