import numpy as np, cv2, json
from pathlib import Path
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')).astype(float)
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')).astype(float)
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
def lc(a): return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
def hs(a):
    a=a/255.0; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0)
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue,np.where(m,sat,0)
lf,cf = lc(final); hf,sf = hs(final); lo,co = lc(orig)
olive=(hf>=45)&(hf<=115)&(sf>0.12)
for cmax in (50,60,70,75,80):
    w=(lf>150)&(cf<cmax)&(~olive)
    band=np.zeros(orig.shape[:2],bool); band[570:790,330:700]=True
    w&=band
    print(f'chroma<{cmax}: visible-pale {int(w.sum())}  release {int((w&rel).sum())}  outside-release {int((w&~rel).sum())}  of outside: guarded {int((w&~rel&grd).sum())}')
# detailed for chroma<70 outside release
w=(lf>150)&(cf<70)&(~olive); band=np.zeros(orig.shape[:2],bool); band[570:790,330:700]=True
w&=band&~rel
print('outside-release visible pale: n=',int(w.sum()),'final med',np.median(final[w],0).round(0),'orig med',np.median(orig[w],0).round(0),'orig chroma med',round(float(np.median(co[w])),1))
n,lab,st,_=cv2.connectedComponentsWithStats((w*255).astype(np.uint8),8)
blobs=sorted([{'px':int(st[i,4]),'bbox':[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]} for i in range(1,n)],key=lambda b:-b['px'])[:10]
print('top blobs',json.dumps(blobs))
for b in blobs[:5]:
    x,y,ww,hh=b['bbox']; m=np.zeros_like(w); m[y:y+hh,x:x+ww]=w[y:y+hh,x:x+ww]
    print('  blob',b['bbox'],'px',b['px'],'final med',np.median(final[m],0).round(0),'orig med',np.median(orig[m],0).round(0))
