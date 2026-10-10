from pathlib import Path
import numpy as np, cv2
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
OUT = REPO/'evidence/p29-collar-adjudication'
final = np.asarray(Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB'))
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
carve = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_carve.png').convert('L'))>0
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
skinp = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_protected_skin.png').convert('L'))>0
x0,y0,x1,y1 = 300,520,720,820
base = final.copy()
ov = base.copy()
ov[rel] = (0.35*ov[rel] + 0.65*np.array([0,220,255])).astype(np.uint8)   # release = cyan
ov[carve] = (0.25*ov[carve] + 0.75*np.array([255,230,0])).astype(np.uint8) # carve = yellow
ov[grd] = (0.55*ov[grd] + 0.45*np.array([60,0,255])).astype(np.uint8)     # guarded = red
ov[skinp] = (0.75*ov[skinp] + 0.25*np.array([0,255,0])).astype(np.uint8)   # protected skin = green
S=3
def rs(a): 
    c=a[y0:y1,x0:x1]; return cv2.resize(c,((x1-x0)*S,(y1-y0)*S),interpolation=cv2.INTER_NEAREST)
panels=[rs(base),rs(ov)]
sheet=np.concatenate(panels,axis=1)
Image.fromarray(sheet).save(OUT/'r8_release_overlay_3x.png')
print('release px',int(rel.sum()),'carve',int(carve.sum()),'guarded',int(grd.sum()))
# how many pale-in-final pixels are outside release, inside band
