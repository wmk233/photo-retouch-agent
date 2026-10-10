"""Four-panel verdict: original | final | final+release/carve | final+guarded."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')

def load(n):
    return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/n).convert('L')) > 127
release = load('collar_repair_mask.png')
carve   = load('collar_repair_carve.png')
guard   = load('collar_repair_guarded_pale.png')

x0, y0, x1, y1 = 380, 545, 680, 800
sc = 5
W, H = (x1-x0)*sc, (y1-y0)*sc
base = np.asarray(final).astype(np.float32)

def small(im):
    return im.crop((x0,y0,x1,y1)).resize((W,H), Image.LANCZOS)

def tinted(masks, colours, alpha=0.55):
    a = base.copy()
    for m, col in zip(masks, colours):
        sub = m[y0:y1, x0:x1]
        blk = a[y0:y1, x0:x1]
        blk[sub] = blk[sub]*(1-alpha) + np.array(col, np.float32)*alpha
    return Image.fromarray(a.astype(np.uint8)).crop((x0,y0,x1,y1)).resize((W,H), Image.LANCZOS)

panels = [
    (small(orig),  'A original'),
    (small(final), 'B final'),
    (tinted([release, carve], [(255,40,40),(0,255,0)]), 'C final + release(red) / carve(green)'),
    (tinted([guard], [(40,90,255)]),                    'D final + guarded(blue)'),
]
sheet = Image.new('RGB', (W*2+6, H*2+6), (20,20,20))
for i,(p,lab) in enumerate(panels):
    x = (i%2)*(W+6); y = (i//2)*(H+6)
    sheet.paste(p, (x,y))
    d = ImageDraw.Draw(sheet)
    d.rectangle([x,y,x+W-1,y+H-1], outline=(200,200,200))
    d.rectangle([x, y, x+len(lab)*7+10, y+16], fill=(0,0,0))
    d.text((x+4, y+3), lab, fill=(255,255,0))
sheet.save(OUT/'verdict_panel_5x.png')
print('wrote', OUT/'verdict_panel_5x.png', sheet.size)
print('release px in crop', int(release[y0:y1,x0:x1].sum()),
      '| carve in crop', int(carve[y0:y1,x0:x1].sum()),
      '| guarded in crop', int(guard[y0:y1,x0:x1].sum()))
