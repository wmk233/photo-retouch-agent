from pathlib import Path
import numpy as np, cv2, json
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB'))
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
carve = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_carve.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
H,W = orig.shape[:2]

def luma_chroma(a):
    a=a.astype(np.float32)
    return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
def hue_sat(a):
    a=a.astype(np.float32)/255.0
    mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.0)
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue, np.where(m,sat,0.0)

ol_f, sl_f = luma_chroma(final)
oh_f, sh_f = hue_sat(final)
ol_o, co_o = luma_chroma(orig)
ch = np.abs(orig.astype(int)-final.astype(int)).max(2)

# "visible pale cloth in the FINAL" -- bright, not the olive blouse (hue 45-115 @ sat>0.12)
olive = (oh_f>=45)&(oh_f<=115)&(sh_f>0.12)
pale = (ol_f>150)&(~olive)
zone = rel.copy(); zone[500:790, 330:700] = True   # collar neighbourhood
pale &= zone
print('FINAL pale-in-collar (luma>150, not olive), px =', int(pale.sum()))
cats = {
 'in_release': pale & rel,
 'in_guarded': pale & grd,
 'outside_release_and_guarded': pale & ~rel & ~grd,
}
for k,m in cats.items():
    n=int(m.sum())
    if n==0: print(k,0); continue
    print(f'{k}: px={n}')
    print('   final RGB med', np.median(final[m],0).round(0), 'luma med', round(float(np.median(ol_f[m])),1), 'chroma med', round(float(np.median(0*ol_f[m]+ (final[m].max(1).astype(float)-final[m].min(1).astype(float)))),1))
    print('   orig  RGB med', np.median(orig[m],0).round(0), 'luma med', round(float(np.median(ol_o[m])),1), 'chroma med', round(float(np.median(co_o[m])),1))
    print('   |orig-final| channel med', round(float(np.median(ch[m])),1), 'p90', round(float(np.percentile(ch[m],90)),1))
    # blob breakdown
    u8=(m*255).astype(np.uint8)
    n2,lab,st,_=cv2.connectedComponentsWithStats(u8,8)
    blobs=sorted([{'px':int(st[i,4]),'bbox':[int(st[i,0]),int(st[i,1]),int(st[i,2]),int(st[i,3])]} for i in range(1,n2)],key=lambda b:-b['px'])[:6]
    print('   top blobs', json.dumps(blobs))

# the specific bright tube: bbox from visual
tb = np.zeros((H,W),bool); tb[575:695, 508:578]=True
print('--- white-tube probe bbox y575-695 x508-578')
print('  final RGB med', np.median(final[tb],0).round(0), 'orig RGB med', np.median(orig[tb],0).round(0))
print('  inside pale & final:', int((pale&tb).sum()), 'of', int(tb.sum()))
print('  pale&tb in release', int((pale&tb&rel).sum()), 'in guarded', int((pale&tb&grd).sum()), 'neither', int((pale&tb&~rel&~grd).sum()))
