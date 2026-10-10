from __future__ import annotations
import sys, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks, FACE_OVAL
from app.domains.portrait.face_lock import (adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour, fill_polygon)
from app.services.native_mask_inpainting import NativeMaskInpainting
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
labels=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w=labels.shape
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
ms=build_semantic_edit_masks(original,lm,labels)
protected=np.asarray(ms.protected)>0; hair=np.asarray(ms.hair)>0; background=labels==0
fw=face_width(lm); cx,cy=(float(v) for v in lm[CHIN])
oval=fill_polygon((h,w),lm[list(FACE_OVAL)])>0
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
rL_arr,rC_arr=LC(np.asarray(ref,dtype=np.float32).reshape(1,1,3))
ref_chroma=float(rC_arr[0,0]); ref_luma=float(rL_arr[0,0])
print("skin reference rgb",np.round(ref,1),"luma",round(ref_luma,1),"chroma(max-min)",round(ref_chroma,1))
win=np.zeros((h,w),bool)
win[int(cy):int(cy+0.62*fw), max(int(cx-0.62*fw),0):min(int(cx+0.75*fw),w)]=True
olive=(fH>=45)&(fH<=115)&(fS>0.12)
pale=(fL>132)&(fC>25)&(fC<72)
guarded=pale&~background&~hair&~olive&skin&win
diff=np.abs(original.astype(np.int16)-final.astype(np.int16)).max(2); unch=diff<10

# unambiguous original knit patch: left shoulder of the sweater, far from skin
knit_patch=np.zeros((h,w),bool); knit_patch[700:820,120:300]=True
knit_patch&=~background&~hair&(labels!=11)
kC=oC[knit_patch]; kL=oL[knit_patch]
print(f"ORIG sweater-body patch n={int(knit_patch.sum())} luma p10/50/90={[round(float(np.percentile(kL,q))) for q in (10,50,90)]} chroma p10/50/90={[round(float(np.percentile(kC,q))) for q in (10,50,90)]}")
# unambiguous original skin: face oval core (left cheek)
skin_patch=np.zeros((h,w),bool); skin_patch[int(cy-150):int(cy-90), int(cx-60):int(cx+60)]=True
skin_patch&=oval&skin&~hair
print(f"ORIG face-skin patch n={int(skin_patch.sum())} luma p10/50/90={[round(float(np.percentile(oL[skin_patch],q))) for q in (10,50,90)]} chroma p10/50/90={[round(float(np.percentile(oC[skin_patch],q))) for q in (10,50,90)]}")
print(f"FINAL same face-skin patch luma={round(float(np.median(fL[skin_patch])))} chroma={round(float(np.median(fC[skin_patch])))}")
# final neck skin: bright warm region under chin, not guarded, not release
neck=np.zeros((h,w),bool); neck[int(cy+20):int(cy+70), int(cx+8):int(cx+32)]=True
neck&=skin&~pale&~hair
print(f"FINAL neck-skin sample n={int(neck.sum())} rgb={np.round(final[neck].mean(0)).astype(int)} luma={round(float(np.median(fL[neck])))} chroma={round(float(np.median(fC[neck])))}")

# guarded breakdown
gC_o=oC[guarded]; gC_f=fC[guarded]
print("guarded oC p10/50/90:",[round(float(np.percentile(gC_o,q))) for q in (10,50,90)])
print("guarded fC p10/50/90:",[round(float(np.percentile(gC_f,q))) for q in (10,50,90)])
print("guarded fL p10/50/90:",[round(float(np.percentile(fL[guarded],q))) for q in (10,50,90)])
print("guarded unchanged frac",round(float(unch[guarded].mean()),3))
print("guarded & oval",int((guarded&oval).sum()),"&hair",int((guarded&hair).sum()),"&prot",int((guarded&protected).sum()))

# evidence test: "sweater's own tone in the original" = oC <= sweater p95
p95=float(np.percentile(kC,95))
print("sweater oC p95 =",round(p95,1))
cand=guarded&unch&(oC<=p95)&~oval
print("cand(unchanged & oC<=p95 & ~oval) px",int(cand.sum()),"of",int(guarded.sum()))
# of those, how many are also NOT her final skin tone (fC < 0.8*ref_chroma)
cand2=cand&(fC<0.8*ref_chroma)
print("cand2 (& final not skin chroma) px",int(cand2.sum()))
# residual: guarded not released
rest=guarded&~cand2
print("retained guarded px",int(rest.sum()))
n,lab,st,_=cv2.connectedComponentsWithStats(rest.astype(np.uint8),8)
rows=sorted([{"px":int(st[i,4]),"bbox":[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])],"ORIG":[int(v) for v in np.round(original[lab==i].mean(0))],"FIN":[int(v) for v in np.round(final[lab==i].mean(0))]} for i in range(1,n) if st[i,4]>=40],key=lambda r:-r["px"])
print("retained blobs>=40:",json.dumps(rows[:8]))
