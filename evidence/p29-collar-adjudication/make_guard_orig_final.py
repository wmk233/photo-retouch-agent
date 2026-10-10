"""Original vs final with the guarded region outlined, plus a hue/chroma swatch strip."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
guard = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L')) > 127
release = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L')) > 127

x0,y0,x1,y1 = 380,545,680,800
sc = 6
W,H = (x1-x0)*sc,(y1-y0)*sc

def panel(im, masks_colours, label):
    p = im.crop((x0,y0,x1,y1)).resize((W,H), Image.LANCZOS)
    d = ImageDraw.Draw(p)
    for m,col in masks_colours:
        sub = m[y0:y1, x0:x1]
        edge = sub & ~np.pad(sub,1)[1:-1,1:-1]
        ys,xs = np.nonzero(edge)
        for y,x in zip(ys,xs):
            d.rectangle([x*sc,y*sc,x*sc+sc-1,y*sc+sc-1], fill=col)
    d.rectangle([0,0,len(label)*7+10,16], fill=(0,0,0))
    d.text((4,3), label, fill=(255,255,0))
    return p

pa = panel(orig,  [(guard,(0,120,255)),(release,(255,40,40))], 'ORIGINAL  blue=guarded red=release')
pb = panel(final, [(guard,(0,120,255)),(release,(255,40,40))], 'FINAL  blue=guarded red=release')
sheet = Image.new('RGB',(W*2+6,H),(20,20,20))
sheet.paste(pa,(0,0)); sheet.paste(pb,(W+6,0))
sheet.save(OUT/'guard_orig_final_6x.png')
print('wrote', sheet.size)

# swatch strip: median colours
def med(m, arr, tag):
    import numpy as _n
    p = arr[m]
    print(f'{tag}: RGB med {_n.median(p,axis=0).round(1)}')
F = np.asarray(final).astype(np.float32); O = np.asarray(orig).astype(np.float32)
med(guard, F, 'final guarded'); med(guard, O, 'orig  guarded')
med(release, F, 'final release'); med(release, O, 'orig  release')
# neck skin strip just under the chin, final
import json
chin = (486, 594)
nk = np.zeros(F.shape[:2], bool); nk[600:640, 455:520] = True
med(nk, F, 'final sample under chin (x455-520,y600-640)')
med(nk, O, 'orig  sample under chin')
