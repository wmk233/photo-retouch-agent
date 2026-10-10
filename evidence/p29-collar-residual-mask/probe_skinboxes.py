import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
d=np.load(f"{WORK}/_masks.npy",allow_pickle=True).item()
sample=np.load(f"{WORK}/_sample.npy"); comm=np.load(f"{WORK}/_committed.npy")
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
oc=orig.max(2)-orig.min(2); ol=0.299*orig[:,:,0]+0.587*orig[:,:,1]+0.114*orig[:,:,2]
skin=d['skin']
H,W=sample.shape
def box(x0,y0,w,h):
    m=np.zeros((H,W),bool); m[y0:y0+h,x0:x0+w]=True; return m
A=box(446,570,84,23); B=box(479,618,31,31)
for nm,m in [("A[446,570,84,23]",A),("B[479,618,31,31]",B)]:
    print(f"{nm}: n={int(m.sum())}  origRGB={np.median(orig[m],0).astype(int).tolist()} luma={np.median(ol[m]):.0f} chroma={np.median(oc[m]):.0f}  skin_model_frac={ (m&skin).sum()/m.sum():.2f}")
    sm=sample&m
    if sm.sum():
        rgb=np.median(orig[sm],0).astype(int).tolist(); lu=float(np.median(ol[sm])); ch=float(np.median(oc[sm]))
        print("   sample-release inside: %d  its origRGB=%s luma=%.0f chroma=%.0f"%(int(sm.sum()),rgb,lu,ch))
        ys,xs=np.where(sm); print("      its bbox=[%d,%d,%d,%d]"%(xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1))
    else:
        print("   sample-release inside: 0")

x0,y0,x1,y1=430,555,585,665
crop_o=orig[y0:y1,x0:x1].astype(np.uint8).copy()
crop_f=fin[y0:y1,x0:x1].astype(np.uint8).copy()
for name,crop in [("orig",crop_o),("fin",crop_f)]:
    g=crop.copy()
    for (bx,by,bw,bh,col) in [(446,570,84,23,(255,0,0)),(479,618,31,31,(0,0,255))]:
        cv2.rectangle(g,(bx-x0,by-y0),(bx-x0+bw-1,by-y0+bh-1),col,1)
    Image.fromarray(g).resize((g.shape[1]*10,g.shape[0]*10),Image.NEAREST).save(f"{WORK}/zz_skintest_{name}_10x.png")
print("wrote zz_skintest_orig_10x.png / zz_skintest_fin_10x.png")
