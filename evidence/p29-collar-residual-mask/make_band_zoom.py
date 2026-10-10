import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
d=np.load(f"{WORK}/_masks.npy",allow_pickle=True).item()
sample=np.load(f"{WORK}/_sample.npy"); comm=np.load(f"{WORK}/_committed.npy")
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
olive=d['olive']; window=d['window']
fl=0.299*fin[:,:,0]+0.587*fin[:,:,1]+0.114*fin[:,:,2]; fc=fin.max(2)-fin.min(2)
vis=(fl>140)&(fc<55)&~olive&window
unconf=vis&~sample&~comm
# add the unreleased part of the block (protected) for context
blob=np.load(f"{WORK}/_blob.npy")

x0,y0,x1,y1=495,585,590,705
def panel(img,outline_mask_list,title_marks):
    c=img[y0:y1,x0:x1].astype(np.uint8).copy()
    for m,col in outline_mask_list:
        mm=(m[y0:y1,x0:x1]*255).astype(np.uint8)
        cnts,_=cv2.findContours(mm,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c,cnts,-1,col,1)
    return c
o=panel(orig,[],None)
f=panel(fin,[(unconf,(255,0,255))],None)
gap=np.full((y1-y0,6,3),255,np.uint8)
comb=np.concatenate([o,gap,f],1)
Image.fromarray(comb).resize((comb.shape[1]*9,comb.shape[0]*9),Image.NEAREST).save(f"{WORK}/zz_band_zoom_9x.png")
print("crop x[%d:%d] y[%d:%d]  scale 9x  left=ORIGINAL right=FINAL(magenta=unconfirmed)"%(x0,x1,y0,y1))
# numeric anchors at this crop
print("reference patches (ORIGINAL RGB / chroma):")
for nm,(px,py) in {"known neck skin (479,640)":(479,640),"band left (512,640)":(512,640),"band core (528,640)":(528,640),"band right (548,640)":(548,640),"known knit fold (570,640)":(570,640),"known green blouse (600,690)":(600,690)}.items():
    p=orig[py-3:py+4,px-3:px+4].reshape(-1,3)
    print("  %-26s ORIG rgb=%s chroma=%.0f | FINAL rgb=%s chroma=%.0f"%(nm,np.median(p,0).astype(int).tolist(),np.median(p.max(1)-p.min(1)),np.median(fin[py-3:py+4,px-3:px+4].reshape(-1,3),0).astype(int).tolist(),np.median((fin[py-3:py+4,px-3:px+4]).reshape(-1,3).max(1)-fin[py-3:py+4,px-3:px+4].reshape(-1,3).min(1))))
