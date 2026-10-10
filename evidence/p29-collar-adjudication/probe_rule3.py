from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
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
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
gp=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png"))>127
oL,oC=LC(o); fL,fC=LC(f); hC,sC=HS(o)

# --- pale mass in the FINAL shared with the approved release strip ---
pale=(fL>145)&(fC<72)
u8=(pale*255).astype(np.uint8); u8=cv2.morphologyEx(u8,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,ll,st,_=cv2.connectedComponentsWithStats(u8,8)
mass=np.zeros_like(pale)
for i in range(1,n):
    m=ll==i
    if (m&rel).sum()>200 and (m&gp).sum()>200: mass|=m
dist=cv2.distanceTransform((~(oC>=70)).astype(np.uint8),cv2.DIST_L2,3)

C_KEEP=60.0; L_BRIGHT=145.0; D_GUARD=2.0
ev_mass = gp & mass
ev_tone = gp & (oC<C_KEEP) & (oL>L_BRIGHT)
ev_far  = gp & (dist>D_GUARD)
release = ev_mass & ev_tone & ev_far
keep = gp & ~release
print("gp",int(gp.sum()))
print(" ev_mass",int(ev_mass.sum())," ev_tone",int(ev_tone.sum())," ev_far",int(ev_far.sum()))
print(" RELEASE",int(release.sum())," KEEP",int(keep.sum()))
def blobs(m,tag):
    u=np.zeros(m.shape,np.uint8); u[m]=255
    nn,_,s,_=cv2.connectedComponentsWithStats(u,8)
    b=sorted([(int(s[i,4]),[int(s[i,0]),int(s[i,1]),int(s[i,2]),int(s[i,3])]) for i in range(1,nn)],reverse=True)
    print(f"  {tag} blobs:",b[:6])
blobs(release,"release"); blobs(keep,"keep")
# composition of kept pixels
print("--- kept composition ---")
print("  oC>=60:",int((keep&(oC>=60)).sum()), " oL<=145:",int((keep&(oL<=145)).sum()),
      " not-in-mass:",int((keep&~mass).sum()), " near-skin<=2:",int((keep&(dist<=2)).sum()))
print("  kept orig C p10/50/90:",[round(float(np.percentile(oC[keep],q)),1) for q in (10,50,90)])
print("  kept orig L p10/50/90:",[round(float(np.percentile(oL[keep],q)),1) for q in (10,50,90)])
print("  kept orig hue p10/50/90:",[round(float(np.percentile(hC[keep],q)),1) for q in (10,50,90)])
print("  rel  orig hue p10/50/90:",[round(float(np.percentile(hC[release],q)),1) for q in (10,50,90)])
print("  rel  orig C p10/50/90:",[round(float(np.percentile(oC[release],q)),1) for q in (10,50,90)])
print("  rel  orig L p10/50/90:",[round(float(np.percentile(oL[release],q)),1) for q in (10,50,90)])
# pale-unchanged residual coverage after release
d=np.linalg.norm(o.astype(np.float32)-f.astype(np.float32),axis=2)
win=np.zeros(o.shape[:2],bool); win[560:745,415:625]=True
resid=(fL>145)&(fC<72)&(d<12)&win
print("--- pale-unchanged residual in window:",int(resid.sum()),
      " covered(old+new):",int((resid&(rel|release)).sum()),
      " left:",int((resid&~(rel|release)).sum()))
left=resid&~(rel|release)
u=np.zeros(o.shape[:2],np.uint8); u[left]=255
nn,_,s,_=cv2.connectedComponentsWithStats(u,8)
print("  left blobs:",sorted([(int(s[i,4]),[int(s[i,0]),int(s[i,1]),int(s[i,2]),int(s[i,3])]) for i in range(1,nn)],reverse=True)[:6])
# overlay image (original crop) -----------------------------------------------
x0,y0,x1,y1=410,555,640,760
crop=o[y0:y1,x0:x1].copy()
ov=crop.copy()
ov[release[y0:y1,x0:x1]]=(1,0,1)     # magenta fill = new release
ov[keep[y0:y1,x0:x1]]=(0,0,1)        # blue fill = kept guarded
gp_final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))[y0:y1,x0:x1]
rel_final=gp_final.copy()
# outline old release in green on the ORIGINAL panel
cont,_=cv2.findContours(((rel[y0:y1,x0:x1])*255).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ov,cont,-1,(0,255,0),1)
cont2,_=cv2.findContours(((release[y0:y1,x0:x1])*255).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ov,cont2,-1,(255,255,255),1)
panels=[]
for im,t in ((crop,"ORIGINAL"),(ov,"ADJUDICATION"),(rel_final,"FINAL")):
    p=im.copy()
    cv2.putText(p,t,(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
    panels.append(p)
S=5
big=np.concatenate([cv2.resize(p,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST) for p in panels],axis=1)
Image.fromarray(big).save("v2_adjudication_5x.png")
print("wrote v2_adjudication_5x.png", big.shape)
