import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
def load(n): return np.asarray(Image.open(REPO/'evidence/p29-collar-repair'/n).convert('L'))>127
release=load('collar_repair_mask.png'); carve=load('collar_repair_carve.png'); guard=load('collar_repair_guarded_pale.png')
x0,y0,x1,y1 = 380,545,680,800
sc=6; W,H=(x1-x0)*sc,(y1-y0)*sc
F=np.asarray(final).astype(np.float32)
def tinted(masks, colours, alpha=0.8):
    a=F.copy()
    for m,col in zip(masks,colours):
        blk=a[y0:y1,x0:x1]; sub=m[y0:y1,x0:x1]
        blk[sub]=blk[sub]*(1-alpha)+np.array(col,np.float32)*alpha
    return Image.fromarray(a.astype(np.uint8)).crop((x0,y0,x1,y1)).resize((W,H),Image.NEAREST)
def plain(im):
    return im.crop((x0,y0,x1,y1)).resize((W,H),Image.LANCZOS)
panels=[
 (plain(orig),'A original'),
 (plain(final),'B final'),
 (tinted([release,carve],[(255,0,0),(0,255,0)]),'C final + RELEASE(red)/carve(green)'),
 (tinted([guard],[(0,110,255)]),'D final + GUARDED(blue)'),
]
sheet=Image.new('RGB',(W*2+6,H*2+6),(20,20,20))
for i,(p,lab) in enumerate(panels):
    x=(i%2)*(W+6); y=(i//2)*(H+6); sheet.paste(p,(x,y))
    d=ImageDraw.Draw(sheet); d.rectangle([x,y,x+len(lab)*7+10,y+16],fill=(0,0,0)); d.text((x+4,y+3),lab,fill=(255,255,0))
sheet.save(OUT/'solid_tint_panel_6x.png'); print('wrote', sheet.size)
