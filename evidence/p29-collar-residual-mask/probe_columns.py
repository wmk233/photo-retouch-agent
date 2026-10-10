import numpy as np
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
oc=d['oc']
def lc(a):
    l=0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]; c=a.max(2)-a.min(2); return l,c
fl,fc=lc(fin)
print("Per-row median chroma of the ORIGINAL by x-band (blob x514-564). knit to the right, skin to the left")
hdr="y    " + "".join(f"{x:>6d}" for x in range(500,601,10))
print(hdr)
for y in range(578,690,8):
    row=[]
    for x in range(500,601,10):
        seg=orig[y-3:y+4,x:x+10]; row.append(np.median(seg.reshape(-1,3).max(1)-seg.reshape(-1,3).min(1)))
    print(f"{y:4d} " + "".join(f"{v:6.0f}" for v in row), "   <-blob spans x514-564" if y%16==2 else "")
print()
print("Per-row median LUMA of the ORIGINAL by x-band")
print(hdr)
for y in range(578,690,8):
    row=[]
    for x in range(500,601,10):
        seg=orig[y-3:y+4,x:x+10]; row.append(np.median(0.299*seg[:,:,0]+0.587*seg[:,:,1]+0.114*seg[:,:,2]))
    print(f"{y:4d} " + "".join(f"{v:6.0f}" for v in row))
print()
print("References (ORIGINAL): sweater lit chroma 25 / shadow 26 / deep 16 ; released strip 32 ; skin neck 86-89")
print("blob row coverage: for y, blob x-extent")
for y in range(578,690,8):
    xs=np.where(blob[y])[0]
    print(f"  y={y:3d} blob x {xs.min() if xs.size else '-'}..{xs.max() if xs.size else '-'} n={xs.size}")
print()
print("== is there an ORIGINAL collar silhouette EDGE along the blob's left boundary? ==")
print("left-of-blob vs blob-edge vs blob-core median RGB (rows 600-680):")
m_l=np.zeros_like(blob); m_l[600:680,500:514]=True
m_e=np.zeros_like(blob); m_e[600:680,514:524]=True
m_c=np.zeros_like(blob); m_c[600:680,530:560]=True
m_r=np.zeros_like(blob); m_r[600:680,566:590]=True
for nm,m in [("x500-514 (left)",m_l),("x514-524 (edge)",m_e),("x530-560 (core)",m_c),("x566-590 (right/knit->fold)",m_r)]:
    print(f"  {nm:26s} origRGB={np.median(orig[m],0).astype(int).tolist()} chroma={np.median(oc[m]):5.1f} | finRGB={np.median(fin[m],0).astype(int).tolist()} finChroma={np.median(fc[m]):5.1f}")
