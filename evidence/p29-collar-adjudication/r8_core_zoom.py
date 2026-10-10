import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB'))
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
carve = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_carve.png').convert('L'))>0
S=6; x0,y0,x1,y1 = 430,570,600,710
def rs(a): return cv2.resize(a[y0:y1,x0:x1],((x1-x0)*S,(y1-y0)*S),interpolation=cv2.INTER_NEAREST)
def outline(img, m, col):
    mm=(m[y0:y1,x0:x1]*255).astype(np.uint8)
    cnt,_=cv2.findContours(mm,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    c=img.copy()
    cv2.polylines(c,[p*S for p in cnt],True,col,2)
    return c
o=rs(orig); f=rs(final)
o2=outline(o, rel, (0,255,255)); o2=outline(o2, carve, (255,230,0)); o2=outline(o2, grd, (255,0,0))
f2=outline(f, rel, (0,255,255)); f2=outline(f2, carve, (255,230,0)); f2=outline(f2, grd, (255,0,0))
def label(a,t):
    im=Image.fromarray(a); d=ImageDraw.Draw(im); d.rectangle([0,0,220,18],fill=(0,0,0)); d.text((4,4),t,fill=(255,255,0)); return np.asarray(im)
sheet=np.concatenate([label(o,'ORIGINAL  +mask'), np.full((o.shape[0],12,3),30,np.uint8), label(f,'FINAL  +mask')],axis=1)
Image.fromarray(sheet).save(OUT/'r8_core_masked_6x.png'); print(sheet.shape)
