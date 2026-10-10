from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
x0,x1,y0,y1=450,590,595,705
Z=10
o=original[y0:y1,x0:x1]; f=final[y0:y1,x0:x1]
row=np.concatenate([o,f],axis=1)
big=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(REPO/"evidence/p29-collar-adjudication/hi10_orig_final.png")
print("saved hi10_orig_final.png",big.shape,"crop",x0,x1,y0,y1)
