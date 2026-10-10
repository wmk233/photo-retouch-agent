import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
def LC(a):
    a=a.astype(np.float32); return (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2], a.max(2)-a.min(2))
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
noedit=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_no_edit.png"))>127
oL,oC=LC(o); fL,fC=LC(f)
# pale mass in the final, shared with the released band
pale=(fL>145)&(fC<72)
u8=((pale*255).astype(np.uint8))
u8=cv2.morphologyEx(u8,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,ll,st,_=cv2.connectedComponentsWithStats(u8,8)
mass=np.zeros_like(pale)
for i in range(1,n):
    m=ll==i
    if (m&rel).sum()>200 and (m&gp).sum()>200: mass|=m
print("pale-mass component area",int(mass.sum()),"covers gp",int((mass&gp).sum()),"rel",int((mass&rel).sum()))
skinhi=oC>=70
near_skin=cv2.dilate(skinhi.astype(np.uint8),np.ones((5,5),np.uint8))>0
print("--- keep-set composition inside gp ---")
keepC=gp&(oC>=60); keepDim=gp&(oL<=150); keepMass=gp&~mass; keepSkin=gp&near_skin
print(" gp total",int(gp.sum()))
print(" orig C>=60 :",int(keepC.sum()))
print(" orig L<=150:",int(keepDim.sum()))
print(" outside pale mass:",int(keepMass.sum()))
print(" within 2px of orig C>=70 skin:",int(keepSkin.sum()))
relnew=gp&(oC<60)&(oL>150)&mass&~near_skin&~noedit
print(" RELEASE (new):",int(relnew.sum()))
print(" keep (gp - release):",int((gp&~relnew).sum()))
ys,xs=np.where(gp&~relnew)
u8k=np.zeros_like(oL,np.uint8);u8k[gp&~relnew]=255
nn,_,stk,_=cv2.connectedComponentsWithStats(u8k,8)
print(" kept blobs:",sorted([(int(stk[i,4]),[int(stk[i,0]),int(stk[i,1]),int(stk[i,2]),int(stk[i,3])]) for i in range(1,nn)],reverse=True)[:6])
u8r=np.zeros_like(oL,np.uint8);u8r[relnew]=255
nr,_,str_,_=cv2.connectedComponentsWithStats(u8r,8)
print(" release blobs:",sorted([(int(str_[i,4]),[int(str_[i,0]),int(str_[i,1]),int(str_[i,2]),int(str_[i,3])]) for i in range(1,nr)],reverse=True)[:6])
# residual after release
win=np.zeros(o.shape[:2],bool); win[560:745,415:625]=True
resid=(fL>145)&(fC<72)&win&(np.linalg.norm(o.astype(np.float32)-f.astype(np.float32),axis=2)<12)
print("--- pale-unchanged residual in window ---")
print(" total",int(resid.sum()),"covered by release(old+new):",int((resid&(rel|relnew)).sum()),"left:",int((resid&~(rel|relnew)).sum()))
left=resid&~(rel|relnew)
u8l=np.zeros_like(oL,np.uint8);u8l[left]=255
nl,_,stl,_=cv2.connectedComponentsWithStats(u8l,8)
print(" left blobs:",sorted([(int(stl[i,4]),[int(stl[i,0]),int(stl[i,1]),int(stl[i,2]),int(stl[i,3])]) for i in range(1,nl)],reverse=True)[:8])
# total visible pale mass in the window, coverage
vm=(fL>145)&(fC<72)&win
print("--- total FINAL pale-cloth mass in window:",int(vm.sum()),"covered:",int((vm&(rel|relnew)).sum()),"uncovered:",int((vm&~(rel|relnew)).sum()))
un=vm&~(rel|relnew)
u8u=np.zeros_like(oL,np.uint8);u8u[un]=255
nu,_,stu,_=cv2.connectedComponentsWithStats(u8u,8)
print(" uncovered blobs:",sorted([(int(stu[i,4]),[int(stu[i,0]),int(stu[i,1]),int(stu[i,2]),int(stu[i,3])]) for i in range(1,nu)],reverse=True)[:10])
