from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
orig  = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB'))
S=5; x0,y0,x1,y1 = 415,575,600,745
def rs(a): return cv2.resize(a[y0:y1,x0:x1],((x1-x0)*S,(y1-y0)*S),interpolation=cv2.INTER_NEAREST)
o=rs(orig); f=rs(final)
for a,lab in ((o,'ORIGINAL'),(f,'FINAL')):
    ImageDraw.Draw(Image.fromarray(a)).rectangle([0,0,120,16],fill=(0,0,0))
    ImageDraw.Draw(Image.fromarray(a)).text((4,3),lab,fill=(255,255,0))
sheet=np.concatenate([o,np.full((o.shape[0],10,3),30,np.uint8),f],axis=1)
Image.fromarray(sheet).save(OUT/'r8_orig_final_neck_5x.png')
print('ok',sheet.shape)
