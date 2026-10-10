from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
final=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
# guarded core strip + reference patches: knit(release) left, chin skin above
x0,x1,y0,y1=500,585,585,735
Z=9
o=original[y0:y1,x0:x1]; f=final[y0:y1,x0:x1]
row=np.concatenate([o,f],axis=1)
row=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_LANCZOS4)
Image.fromarray(row).save(REPO/"evidence/p29-collar-adjudication/guarded_texture_9x.png")
print("saved",row.shape,"orig left, final right")
