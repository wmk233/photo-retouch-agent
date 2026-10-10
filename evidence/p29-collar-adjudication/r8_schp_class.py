from pathlib import Path
import numpy as np, json
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
lab = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/labels_fresh.png'))
print('labels shape',lab.shape,'dtype',lab.dtype,'uniq',np.unique(lab))
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
skinp = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_protected_skin.png').convert('L'))>0
def dist(m,name):
    v=lab[m]
    n=len(v)
    if n==0: print(name,'empty'); return
    u,c=np.unique(v,return_counts=True)
    print(f'{name}: px={n}', {int(k):round(float(cc)/n,3) for k,cc in zip(u,c)})
dist(rel,'release')
dist(grd,'guarded_pale')
dist(skinp,'protected_skin(in no_edit)')
