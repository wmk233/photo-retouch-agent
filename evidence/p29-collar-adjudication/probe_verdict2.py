"""Guarded-region adjudication with pipeline-derived references."""
import numpy as np, json
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
F = np.asarray(final).astype(np.float32)
O = np.asarray(orig).astype(np.float32)

def chroma(a): return a.max(axis=-1) - a.min(axis=-1)
def luma(a):   return 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]
def load(n):   return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/n).convert('L')) > 127

release = load('collar_repair_mask.png')
guard   = load('collar_repair_guarded_pale.png')
protskin= load('collar_repair_protected_skin.png')
noedit  = load('collar_repair_no_edit.png')

stats = json.loads((REPO/'evidence/p29-collar-repair/collar_repair_stats.json').read_text())
print('stats keys:', [k for k in stats.keys()])
print('skin_reference_colour:', stats.get('skin_reference_colour'),
      ' face_width:', stats.get('face_width'))

def med(mask, arr, tag):
    p = arr[mask]
    print(f'  {tag}: n={len(p)} RGB med {np.median(p,axis=0).round(1)} '
          f'luma {np.percentile(luma(p),50):.0f} chroma {np.percentile(chroma(p),50):.0f}')

print('skin reference (pipeline protected_skin):')
med(protskin, F, 'protected_skin final')
med(protskin, O, 'protected_skin orig')

print('white-knit reference (ORIGINAL release strip = the old pale cloth):')
med(release, O, 'release orig')
med(release, F, 'release final')

print('guarded region:')
med(guard, F, 'guarded final')
med(guard, O, 'guarded orig')

skin = np.median(F[protskin], axis=0)
knit = np.median(O[release], axis=0)
print('skin ref', skin.round(1), ' knit ref', knit.round(1),
      ' ||skin-knit|| =', round(float(np.linalg.norm(skin-knit)),1))

gF, gO = F[guard], O[guard]
for tag, arr in (('final', gF), ('orig', gO)):
    d_skin = np.linalg.norm(arr - skin, axis=1)
    d_knit = np.linalg.norm(arr - knit, axis=1)
    m = d_skin - d_knit
    print(f'  guarded {tag}: closer to knit {int((m>0).sum())}/{len(arr)}  '
          f'margin>2 {int((m>2).sum())}  >10 {int((m>10).sum())}  '
          f'margin p10/50/90 {np.percentile(m,10).round(1)}/{np.percentile(m,50).round(1)}/{np.percentile(m,90).round(1)}')

relF, relO = F[release], O[release]
for tag, arr in (('final', relF), ('orig', relO)):
    d_skin = np.linalg.norm(arr - skin, axis=1)
    d_knit = np.linalg.norm(arr - knit, axis=1)
    m = d_skin - d_knit
    print(f'  release {tag}: closer to knit {int((m>0).sum())}/{len(arr)}  '
          f'margin p50 {np.percentile(m,50).round(1)}')

# how much of the FINAL's pale mass is inside release vs guarded?
pale_f = (luma(F) > 150) & (chroma(F) < 45)
band = np.zeros(F.shape[:2], bool); band[560:790, 330:680] = True
pfb = pale_f & band
print('collar-band pale (luma>150,chroma<45) in FINAL:', int(pfb.sum()),
      ' in release:', int((pfb & release).sum()),
      ' in guarded:', int((pfb & guard).sum()),
      ' in no_edit(uncovered):', int((pfb & noedit).sum()))
