import numpy as np, cv2, json
from pathlib import Path
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')).astype(float)
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')).astype(float)
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
def lc(a): return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
lf,cf=lc(final); lo,co=lc(orig)
w=(lf>180)&(cf<70)
band=np.zeros(orig.shape[:2],bool); band[570:800,330:700]=True
w&=band
print('bright white (final luma>180, chroma<70) in band:', int(w.sum()), 'release', int((w&rel).sum()), 'guarded', int((w&grd).sum()), 'neither', int((w&~rel&~grd).sum()))
print(' final med',np.median(final[w],0).round(0),'orig med',np.median(orig[w],0).round(0))
n,lab,st,_=cv2.connectedComponentsWithStats((w*255).astype(np.uint8),8)
blobs=sorted([{'px':int(st[i,4]),'bbox':[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]} for i in range(1,n)],key=lambda b:-b['px'])[:8]
print(' blobs',json.dumps(blobs))
for b in blobs[:4]:
    x,y,ww,hh=b['bbox']; m=np.zeros_like(w); m[y:y+hh,x:x+ww]=w[y:y+hh,x:x+ww]
    print('   ',b['bbox'],b['px'],'rel',int((m&rel).sum()),'grd',int((m&grd).sum()),'final',np.median(final[m],0).round(0),'orig',np.median(orig[m],0).round(0))
