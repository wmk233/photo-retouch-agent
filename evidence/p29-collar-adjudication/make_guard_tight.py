import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
guard = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L')) > 127
x0,y0,x1,y1 = 427,584,576,736
sc=8; W,H=(x1-x0)*sc,(y1-y0)*sc
def panel(im, outline, label):
    p = im.crop((x0,y0,x1,y1)).resize((W,H), Image.LANCZOS)
    d = ImageDraw.Draw(p)
    if outline:
        sub = guard[y0:y1,x0:x1]; e = sub & ~np.pad(sub,1)[1:-1,1:-1]
        ys,xs=np.nonzero(e)
        for y,x in zip(ys,xs):
            d.rectangle([x*sc+1,y*sc+1,x*sc+sc-2,y*sc+sc-2], fill=(0,140,255))
    d.rectangle([0,0,len(label)*7+10,16], fill=(0,0,0)); d.text((4,3),label,fill=(255,255,0))
    return p
a=panel(orig, True,  'ORIGINAL  (blue = guarded outline)')
b=panel(final,True,  'FINAL     (blue = guarded outline)')
sheet=Image.new('RGB',(W*2+6,H),(20,20,20)); sheet.paste(a,(0,0)); sheet.paste(b,(W+6,0))
sheet.save(OUT/'guard_tight_orig_final_8x.png'); print('wrote', sheet.size)
