import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
H,W=orig.shape[:2]
skin=d['skin']; hair=d['hair']; background=d['background']; oval=d['oval']
change=d['change']; ol=d['ol']; oc=d['oc']; fl=d['fl']; fc=d['fc']

gray=cv2.cvtColor(orig.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
# high-frequency energy: box (5x5) variance of the Laplacian  -> knit loops ring, skin is smooth
lap=cv2.Laplacian(gray,cv2.CV_32F,ksize=3)
hf=cv2.boxFilter(lap*lap,-1,(5,5))
# gradient magnitude too
gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
gm=np.sqrt(gx*gx+gy*gy)

def region_stats(name, m):
    m=m&(change>=0)
    if m.sum()==0: print(f"{name}: empty"); return
    print(f"{name:34s} px={int(m.sum()):6d}  HFsqrt={np.sqrt(hf[m]).mean():5.2f}  |grad|={gm[m].mean():5.2f}  "
          f"origRGB_med={np.median(orig[m],0).astype(int).tolist()}  origChroma={np.median(oc[m]):5.1f}  "
          f"unchanged<10={100*(change[m]<10).mean():5.1f}%")

print("== original-structure texture (HF = rms Laplacian over 5x5; skin is smooth, knit is not) ==")
region_stats("block 1067px [514,578,51,111]", blob)
region_stats("released strip (mask)", np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127)
region_stats("sweater body (orig knit, big)", (ol>150)&(oc<45)&(np.arange(H)[:,None]>760)&(skin==False)&(background==False)&(hair==False))
region_stats("true neck skin A [446,570,530,593]", np.zeros((H,W),bool).__setitem__((slice(570,593),slice(446,530)),True) or np.zeros((H,W),bool)) if False else None
for nm,(x0,y0,x1,y1) in {"true neck skin A":(446,570,530,593),"true neck skin B":(479,618,510,649)}.items():
    m=np.zeros((H,W),bool); m[y0:y1,x0:x1]=True
    region_stats(nm+f" [{x0},{y0},{x1-x0},{y1-y0}]", m)
for nm,(x0,y0,x1,y1) in {"chin skin":(455,555,520,578),"cheek skin":(430,470,470,520)}.items():
    m=np.zeros((H,W),bool); m[y0:y1,x0:x1]=True
    region_stats(nm+f" [{x0},{y0},{x1-x0},{y1-y0}]", m)
m=np.zeros((H,W),bool); m[150:250,150:250]=True; region_stats("background patch [150,150,100,100]", m)

print()
print("== did the renderer repaint the neck? (window = collar band) ==")
cx,cy,fw=486.61,594.19,256.72
win=np.zeros((H,W),bool); win[int(cy):int(cy+0.62*fw),int(cx-0.62*fw):int(cx+0.75*fw)]=True
sk=skin&win
print(f"window px={int(win.sum())}  skin-in-window px={int(sk.sum())}")
print(f"skin-in-window changed(maxch>=10)={100*(change[sk]>=10).mean():5.1f}%  median change={np.median(change[sk]):.0f}")
print(f"skin-in-window & block       px={int((sk&blob).sum())}  changed={100*(change[sk&blob]>=10).mean():5.1f}%")
print(f"skin-in-window & ~block      px={int((sk&~blob).sum())} changed={100*(change[sk&~blob]>=10).mean():5.1f}%")
print(f"changed pixels in window px={int((win&(change>=10)).sum())}  of which skin={int((win&(change>=10)&skin).sum())}")
# where did the renderer CREATE new neck? orig-not-skin & final-looks-like-skin & changed
print(f"new neck-ish (final chroma 45-95 & luma>120 & changed) in window px={int((win&(fc>45)&(fc<95)&(fl>120)&(change>=10)).sum())}")

print()
print("== seam test: is the block boundary a material step in the FINAL? ==")
# boundary ring of the block vs interior, in FINAL tone
ker=np.ones((3,3),np.uint8)
ring=(cv2.dilate(blob.astype(np.uint8),ker)>0)&~blob
inside=blob&(cv2.erode(blob.astype(np.uint8),ker)>0)
outside=(cv2.dilate(blob.astype(np.uint8),np.ones((9,9),np.uint8))>0)&~ring&~blob
for nm,m in {"block interior":inside,"block ring":ring,"just outside block":outside}.items():
    print(f"{nm:22s} px={int(m.sum()):5d} finRGB_med={np.median(fin[m],0).astype(int).tolist()} finChroma_med={np.median(fc[m]):5.1f} finLuma_med={np.median(fl[m]):5.1f}")
