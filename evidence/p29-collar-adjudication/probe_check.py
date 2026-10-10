import numpy as np
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
def LC(a):
    a=a.astype(np.float32)
    return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
orig=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
oL,oC=LC(orig); fL,fC=LC(final)
d=np.linalg.norm(orig.astype(np.float32)-final.astype(np.float32),axis=2)
print("rel px",int(rel.sum()),"gp px",int(gp.sum()))
print("rel: d<12",int((rel&(d<12)).sum()),"fL>145",int((rel&(fL>145)).sum()),"fC<72",int((rel&(fC<72)).sum()))
print("rel: d<12 & fL>145 & fC<72 ->",int((rel&(d<12)&(fL>145)&(fC<72)).sum()))
print("rel fC p10/50/90",[round(float(np.percentile(fC[rel],q)),1) for q in (10,50,90)])
print("rel fL p10/50/90",[round(float(np.percentile(fL[rel],q)),1) for q in (10,50,90)])
print("rel d p10/50/90",[round(float(np.percentile(d[rel],q)),1) for q in (10,50,90)])
print("gp fC p10/50/90",[round(float(np.percentile(fC[gp],q)),1) for q in (10,50,90)])
print("gp d p10/50/90",[round(float(np.percentile(d[gp],q)),1) for q in (10,50,90)])
ys,xs=np.where(rel)
print("rel bbox",xs.min(),ys.min(),xs.max(),ys.max())
u8=np.zeros_like(oL,np.uint8); u8[rel]=255
import cv2
n,ll,st,_=cv2.connectedComponentsWithStats(u8,8)
print("rel comps",[(int(st[i,4]),[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]) for i in range(1,n)])
