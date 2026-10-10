"""Where are the pale pixels in the collar band that are NOT in the release mask?"""
import numpy as np, json
from pathlib import Path
from PIL import Image, ImageDraw
from collections import deque

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
F = np.asarray(final).astype(np.float32); O = np.asarray(orig).astype(np.float32)

def chroma(a): return a.max(axis=-1)-a.min(axis=-1)
def luma(a):   return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
def load(n):   return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/n).convert('L'))>127

release = load('collar_repair_mask.png')
guard   = load('collar_repair_guarded_pale.png')
protskin= load('collar_repair_protected_skin.png')

L = luma(F); C = chroma(F)
band = np.zeros(F.shape[:2], bool); band[560:790, 330:680] = True
pale = (L > 150) & (C < 45) & band
print('band pale total', int(pale.sum()))
for tag, m in (('release', release), ('guarded', guard), ('protected_skin', protskin)):
    print(f'  in {tag}: {int((pale&m).sum())}')
rest = pale & ~release & ~guard
print('  outside release+guarded:', int(rest.sum()))
if rest.sum():
    p = F[rest]
    print('   RGB med', np.median(p,axis=0).round(1),
          'luma p10/50/90', np.percentile(luma(p),10).round(0), np.percentile(luma(p),50).round(0), np.percentile(luma(p),90).round(0),
          'chroma p50/90', np.percentile(chroma(p),50).round(0), np.percentile(chroma(p),90).round(0))
    ys, xs = np.nonzero(rest)
    print('   bbox x', xs.min(), xs.max(), ' y', ys.min(), ys.max())
    # blobs (8-conn, stride-2 for speed)
    seen = np.zeros_like(rest)
    blobs = []
    for y, x in zip(ys, xs):
        if seen[y, x]: continue
        q = deque([(y, x)]); seen[y, x] = True; n = 0
        minx=maxx=x; miny=maxy=y
        while q:
            cy, cx = q.popleft(); n += 1
            minx=min(minx,cx); maxx=max(maxx,cx); miny=min(miny,cy); maxy=max(maxy,cy)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<rest.shape[0] and 0<=nx<rest.shape[1] and rest[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True; q.append((ny,nx))
        blobs.append((n, minx, miny, maxx, maxy))
    blobs.sort(reverse=True)
    for b in blobs[:12]:
        print('   blob px', b[0], 'bbox', b[1:])

# visual: band crop 8x with outlines
x0,y0,x1,y1 = 330,560,680,790
sc = 8
crop = final.crop((x0,y0,x1,y1)).resize(((x1-x0)*sc,(y1-y0)*sc), Image.NEAREST)
d = ImageDraw.Draw(crop)
def outline(mask, colour):
    sub = mask[y0:y1, x0:x1]
    edge = sub & ~np.pad(sub,1)[1:-1,1:-1]
    ys, xs = np.nonzero(edge)
    for y, x in zip(ys, xs):
        d.rectangle([x*sc, y*sc, x*sc+sc-1, y*sc+sc-1], fill=colour)
outline(release, (255,60,60))
outline(guard, (70,110,255))
sub = rest[y0:y1, x0:x1]
ys, xs = np.nonzero(sub)
for y, x in zip(ys, xs):
    d.rectangle([x*sc+2, y*sc+2, x*sc+sc-3, y*sc+sc-3], fill=(255,0,255))
crop.save(OUT/'band_pale_uncovered_8x.png')
print('wrote', OUT/'band_pale_uncovered_8x.png', crop.size)
