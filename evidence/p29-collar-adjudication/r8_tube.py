import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB'))
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
S=8; x0,y0,x1,y1 = 495,565,605,700
def rs(a): return cv2.resize(a[y0:y1,x0:x1],((x1-x0)*S,(y1-y0)*S),interpolation=cv2.INTER_NEAREST)
def outl(img,m,col):
    mm=(m[y0:y1,x0:x1]*255).astype(np.uint8)
    cnt,_=cv2.findContours(mm,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    c=img.copy(); cv2.polylines(c,[p*S for p in cnt],True,col,2); return c
o=outl(outl(rs(orig),rel,(0,255,255)),grd,(255,0,0))
f=outl(outl(rs(final),rel,(0,255,255)),grd,(255,0,0))
def lb(a,t):
    im=Image.fromarray(a); d=ImageDraw.Draw(im); d.rectangle([0,0,300,20],fill=(0,0,0)); d.text((4,5),t,fill=(255,255,0)); return np.asarray(im)
sheet=np.concatenate([lb(o,'ORIG  cyan=release red=guarded'), np.full((o.shape[0],14,3),40,np.uint8), lb(f,'FINAL same')],axis=1)
Image.fromarray(sheet).save(OUT/'r8_tube_8x.png'); print(sheet.shape)
