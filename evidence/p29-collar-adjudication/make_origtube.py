import numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
o=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
f=np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
x0,x1,y0,y1=470,585,565,700
Z=14
def crop(a): return a[y0:y1,x0:x1].copy()
row=np.concatenate([crop(o),crop(f)],axis=1)
big=cv2.resize(row,(row.shape[1]*Z,row.shape[0]*Z),interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(REPO/"evidence/p29-collar-adjudication/orig_final_tube_14x.png")
print("saved",big.shape)
