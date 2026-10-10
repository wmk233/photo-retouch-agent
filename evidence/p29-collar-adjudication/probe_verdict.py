"""Decisive verdict probe: is the guarded pale mass skin, or leftover white knit?"""
import numpy as np
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
print('final', final.size, 'orig', orig.size)

F = np.asarray(final).astype(np.float32)
O = np.asarray(orig).astype(np.float32)
if O.shape[:2] != F.shape[:2]:
    print('size mismatch -> LANCZOS resize orig to final frame')
    O = np.asarray(orig.resize(final.size, Image.LANCZOS)).astype(np.float32)

def chroma(a):
    return a.max(axis=-1) - a.min(axis=-1)
def luma(a):
    return 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]

def load(name):
    return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/name).convert('L')) > 127

release = load('collar_repair_mask.png')
carve   = load('collar_repair_carve.png')
guard   = load('collar_repair_guarded_pale.png')
noedit  = load('collar_repair_no_edit.png')

def report(label, mask):
    n = int(mask.sum())
    if n == 0:
        print(label, ': empty'); return
    f = F[mask]; o = O[mask]
    c, l = chroma(f), luma(f)
    co, lo = chroma(o), luma(o)
    nw = int(((l > 160) & (c < 20)).sum())
    print(f'{label}: n={n}')
    print(f'   final RGB med {np.median(f,axis=0).round(1)}  luma p10/50/90 '
          f'{np.percentile(l,10):.0f}/{np.percentile(l,50):.0f}/{np.percentile(l,90):.0f}  '
          f'chroma p10/50/90 {np.percentile(c,10):.0f}/{np.percentile(c,50):.0f}/{np.percentile(c,90):.0f}')
    print(f'   orig  RGB med {np.median(o,axis=0).round(1)}  luma p50 {np.percentile(lo,50):.0f}  '
          f'chroma p50 {np.percentile(co,50):.0f}')
    print(f'   neutral-white in FINAL (luma>160 & chroma<20) = {nw} px ({100*nw/n:.2f}%)')

report('release', release)
report('carve', carve)
report('guarded', guard)
report('no_edit', noedit)

print('--- references (final) ---')
refs = {
  'cheek(skin)':  (250,540,320,590),
  'chin(skin)':   (470,585,520,615),
  'olive_blouse': (300,900,380,960),
  'hair':         (600,520,660,560),
}
skin_refs = []
for name,(x0,y0,x1,y1) in refs.items():
    p = F[y0:y1, x0:x1].reshape(-1,3)
    l, c = luma(p), chroma(p)
    print(f'{name}: RGB med {np.median(p,axis=0).round(1)} luma {np.percentile(l,50):.0f} '
          f'chroma {np.percentile(c,50):.0f}')
    if 'skin' in name:
        skin_refs.append(np.median(p,axis=0))
skin = np.mean(skin_refs, axis=0)

knit_sel = (luma(O) > 160) & (chroma(O) < 25)
band = np.zeros(F.shape[:2], bool); band[560:790, 380:640] = True
knit_band = knit_sel & band
print('orig neutral-white knit inside collar band (x380-640,y560-790):', int(knit_band.sum()), 'px')
knit = np.median(O[knit_band], axis=0) if knit_band.sum() else None
print('   knit ref RGB med', None if knit is None else knit.round(1))

if knit is not None:
    gf = F[guard]
    d_skin = np.linalg.norm(gf - skin, axis=1)
    d_knit = np.linalg.norm(gf - knit, axis=1)
    closer_knit = int((d_knit < d_skin).sum())
    margin = d_skin - d_knit
    print(f'guarded vs refs: closer to knit {closer_knit}/{len(gf)}  '
          f'margin>2.0 {(margin>2.0).sum()}  margin>10 {(margin>10).sum()}')
    print('   margin p10/50/90', np.percentile(margin,10).round(1),
          np.percentile(margin,50).round(1), np.percentile(margin,90).round(1))
