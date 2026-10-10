"""Residual measure calibrated on the released cream-knit strip, not chroma gates."""
import numpy as np
from pathlib import Path
from PIL import Image
from collections import deque

REPO = Path(__file__).resolve().parents[2]
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
F = np.asarray(final).astype(np.float32); O = np.asarray(orig).astype(np.float32)
def load(n): return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/n).convert('L'))>127
release=load('collar_repair_mask.png'); carve=load('collar_repair_carve.png'); guard=load('collar_repair_guarded_pale.png')

rel_f = F[release]; mu = np.median(rel_f, axis=0); sd = rel_f.std(axis=0) + 1e-6
mu_o = np.median(O[release], axis=0); sd_o = O[release].std(axis=0) + 1e-6
print('release final median', mu.round(1), 'sd', sd.round(1))
print('release orig  median', mu_o.round(1), 'sd', sd_o.round(1))

band = np.zeros(F.shape[:2], bool); band[570:790, 400:700] = True
z_f = np.abs((F - mu)/sd).max(axis=-1)          # per-channel robust-ish
z_o = np.abs((O - mu_o)/sd_o).max(axis=-1)      # is THIS pixel knit in the ORIGINAL?

for thr in (1.5, 2.0, 2.5):
    kn = (z_f < thr) & band
    print(f'-- final knit-like (z<{thr}) in band: {int(kn.sum())}'
          f'  | release {int((kn&release).sum())}'
          f'  | carve {int((kn&carve).sum())}'
          f'  | guarded {int((kn&guard).sum())}'
          f'  | outside both {int((kn&~release&~guard&~carve).sum())}')

kn_o = (z_o < 2.0) & band
print('ORIGINAL knit-like in band (z<2.0):', int(kn_o.sum()))
print('  of those, final still knit-like:', int((kn_o & (z_f < 2.0)).sum()))
print('  of those, in release:', int((kn_o & release).sum()),
      ' guarded:', int((kn_o & guard).sum()),
      ' outside both:', int((kn_o & ~release & ~guard & ~carve).sum()))

kn = (z_f < 2.0) & band
outside = kn & ~release & ~guard & ~carve
ys, xs = np.nonzero(outside)
if len(ys):
    print('outside-both bbox x', xs.min(), xs.max(), 'y', ys.min(), ys.max())
    seen = np.zeros_like(outside); blobs=[]
    for y,x in zip(ys,xs):
        if seen[y,x]: continue
        q=deque([(y,x)]); seen[y,x]=True; n=0; b=[x,y,x,y]
        while q:
            cy,cx=q.popleft(); n+=1
            b[0]=min(b[0],cx); b[2]=max(b[2],cx); b[1]=min(b[1],cy); b[3]=max(b[3],cy)
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<outside.shape[0] and 0<=nx<outside.shape[1] and outside[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True; q.append((ny,nx))
        blobs.append((n,)+tuple(b))
    blobs.sort(reverse=True)
    for bl in blobs[:10]:
        p = F[bl[2]:bl[4]+1, bl[1]:bl[3]+1].reshape(-1,3)
        print('  blob px', bl[0], 'bbox', bl[1:], 'RGB med', np.median(p,axis=0).round(0))
