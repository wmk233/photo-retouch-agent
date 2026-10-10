import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
d=np.load(f"{WORK}/_masks.npy",allow_pickle=True).item()
sample=np.load(f"{WORK}/_sample.npy"); comm=np.load(f"{WORK}/_committed.npy")
blob=np.load(f"{WORK}/_blob.npy")
skin,hair,bg,oval,prot,olive=d['skin'],d['hair'],d['background'],d['oval'],d['protected'],d['olive']
fl,fc=d['fl'],d['fc']
H,W=sample.shape
chin_y=594.19
def box(x0,y0,x1,y1):
    m=np.zeros((H,W),bool); m[y0:y1,x0:x1]=True; return m
regions={"true neck skin A [446,570,84,23]":box(446,570,530,593),
         "true neck skin B [479,618,31,31]":box(479,618,510,649),
         "chin/jaw     [455,555,65,23]":box(455,555,520,578),
         "cheek        [430,470,40,50]":box(430,470,470,520)}
print("== PROTECTION CHECK: how much of the release lands on each protected thing ==")
print(f"release px                 {int(sample.sum())}")
print(f"  on face oval             {int((sample&oval).sum())}")
print(f"  on hair                  {int((sample&hair).sum())}")
print(f"  on background            {int((sample&bg).sum())}")
print(f"  on new olive blouse      {int((sample&olive).sum())}")
print(f"  above the chin line y<594{int((sample[:(594),:]).sum())}   (face side of the jaw)")
for nm,m in regions.items():
    print(f"  on {nm:32s} {int((sample&m).sum())} / {int(m.sum())}")
prot_only=prot&~oval&~(np.arange(H)[:,None]<chin_y)
print(f"  on identity/throat guard {int((sample&prot).sum())}  (of which below chin & off-oval: {int((sample&prot_only).sum())})")
print()
print("== the release geometry ==")
for nm,m in [("committed release (prior rounds)",comm),("sample release (this round)",sample)]:
    ys,xs=np.where(m); print(f"  {nm:34s} px={int(m.sum()):5d} bbox=[{xs.min()},{ys.min()},{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}]")
n,_,st,_=cv2.connectedComponentsWithStats((sample*255).astype(np.uint8),8)
print("  sample release blobs:",sorted([ (int(st[i,4]),[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]) for i in range(1,n)],reverse=True)[:5])
print()
print("== COVERAGE of the visible residual (final luma>140 & chroma<55, not olive, in the collar band) ==")
vis=(fl>140)&(fc<55)&~olive&d['window']
print(f"  visible pale residual in band      {int(vis.sum())}")
print(f"  covered by committed release       {int((vis&comm).sum())}")
print(f"  covered by sample release          {int((vis&sample).sum())}")
print(f"  still NOT covered (unconfirmed)    {int((vis&~sample).sum())}")
print(f"  of the block 1067px: released {int((blob&sample).sum())} / still protected {int((blob&~sample).sum())}")
print(f"  block px that are NOT visibly pale in the final: {int((blob&~vis).sum())}")
print()
print("== the two sizes of the residual that the previous round reported ==")
print("  (prior round) visible pale cloth 3186px, outside edit region 1748px, biggest blob 1067px")
out=(vis&~comm)
print(f"  recomputed outside committed release {int(out.sum())}px  -> sample now covers {int((out&sample).sum())}px ({100*(out&sample).sum()/max(out.sum(),1):.1f}%)")
