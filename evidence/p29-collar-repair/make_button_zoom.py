from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
mask = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>0
ov = final.copy(); ov[mask]=[255,0,0]
y0,y1,x0,x1 = 660,745,470,600
def z(img):
    c = img[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(c).resize((c.shape[1]*6, c.shape[0]*6), Image.NEAREST))
row = np.concatenate([z(final), z(ov)], axis=1)
Image.fromarray(row).save(REPO/"evidence/p29-collar-repair/button_zoom6x.png")
print(row.shape)
