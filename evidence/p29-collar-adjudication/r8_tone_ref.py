from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
orig = np.asarray(Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')).astype(float)
final= np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')).astype(float)
def lc(a):
    return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
def patch(x0,y0,x1,y1,img,name):
    p=img[y0:y1,x0:x1].reshape(-1,3)
    l,c=lc(p.reshape(1,-1,3))
    print(f'{name}: RGB med {np.median(p,0).round(0)} luma {np.median(l):.1f} chroma {np.median(c):.1f}  (n={len(p)})')
print('=== ORIGINAL patches')
patch(300,420,340,460,orig,'cheek(face skin)')
patch(600,640,640,690,orig,'neck right of neckline (skin?)')
patch(560,690,600,720,orig,'below-right of neck (skin?)')
patch(470,600,540,660,orig,'guarded band core')
patch(170,760,260,860,orig,'white knit sleeve (cloth ref)')
print('=== FINAL patches')
patch(600,640,640,690,final,'neck right (skin?)')
patch(470,600,540,660,final,'guarded band core')
patch(640,700,700,760,final,'green blouse')
patch(170,760,260,860,final,'white knit sleeve (cloth ref)')
