from pathlib import Path
import numpy as np, cv2
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB'))
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
ys,xs = np.where(grd); print('guarded bbox', xs.min(), ys.min(), xs.max(), ys.max())
# orig tone at guarded
o = orig[grd].astype(float); f = final[grd].astype(float)
for name,a in (('orig',o),('final',f)):
    print(name,'median',np.median(a,0),'p10',np.percentile(a,10,0),'p90',np.percentile(a,90,0))
ol = 0.299*a[:,0] if False else None
def lc(a):
    l = 0.299*a[:,0]+0.587*a[:,1]+0.114*a[:,2]; c = a.max(1)-a.min(1); return l,c
lo,co = lc(o); lf,cf = lc(f)
print('orig luma med',np.median(lo),'chroma med',np.median(co))
print('final luma med',np.median(lf),'chroma med',np.median(cf))
ch = np.abs(orig.astype(int)-final.astype(int)).max(2)
print('change med at guarded',np.median(ch[grd]), 'p90', np.percentile(ch[grd],90))
