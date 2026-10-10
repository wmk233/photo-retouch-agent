from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
mask = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))
orig = np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
y0,y1,x0,x1 = 580,730,420,650
def z(img):
    c = img[y0:y1, x0:x1]
    return Image.fromarray(c).resize((c.shape[1]*4, c.shape[0]*4), Image.LANCZOS)
ov = final.copy(); ov[mask>0] = [255,0,0]
row = np.concatenate([np.asarray(z(orig)), np.asarray(z(final)), np.asarray(z(ov))], axis=1)
Image.fromarray(row).save(REPO/"evidence/p29-collar-repair/zoom4x_orig_final_mask.png")
print(row.shape)
