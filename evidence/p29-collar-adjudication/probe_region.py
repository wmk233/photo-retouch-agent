"""Adjudication probe: original vs final vs SCHP labels around the guarded band."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
labels = np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/labels_fresh.png"))
print("shapes", orig.shape, final.shape, labels.shape)
print("label values present:", np.unique(labels))

# guarded_pale bbox (x0,y0,x1,y1) reported: [437,594,566,726]
x0, y0, x1, y1 = 437, 594, 566, 726
m = 45
X0, Y0 = max(x0 - m, 0), max(y0 - m, 0)
X1, Y1 = min(x1 + m, orig.shape[1]), min(y1 + m, orig.shape[0])
print("crop", X0, Y0, X1, Y1, "size", X1 - X0, Y1 - Y0)

# colorize labels
PALETTE = {
    0:(0,0,0),1:(200,0,0),2:(255,200,0),3:(0,200,255),4:(0,120,255),5:(180,0,255),
    6:(0,255,120),7:(255,0,180),8:(128,128,0),9:(255,128,0),10:(128,0,0),
    11:(0,255,0),12:(120,255,255),13:(60,120,255),14:(255,255,0),15:(255,0,0),
    16:(0,100,0),17:(100,100,255),
}
rgb = np.zeros((*labels.shape, 3), np.uint8)
for k, v in PALETTE.items():
    rgb[labels == k] = v
lab_img = Image.fromarray(rgb)

Z = 5
def crop_up(arr):
    c = arr[Y0:Y1, X0:X1]
    im = Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width * Z, im.height * Z), Image.NEAREST)

p_orig = crop_up(orig)
p_final = crop_up(final)
p_lab = crop_up(np.asarray(lab_img))

# overlay: mark guarded band + release on original for context
ov = orig.copy()
# load masks
rel = np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png")) > 127
gp = np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_guarded_pale.png")) > 127
ov[rel] = (255, 0, 0)
ov[gp] = (255, 220, 0)
p_ov = crop_up(ov)

W = p_orig.width
H = p_orig.height
canvas = Image.new("RGB", (W * 4 + 3 * 8, H + 20), (30, 30, 30))
for i, (im, t) in enumerate([(p_orig, "ORIGINAL"), (p_final, "FINAL"), (p_ov, "ORIG rel=red guarded=ylw"), (p_lab, "SCHP labels")]):
    canvas.paste(im, (i * (W + 8), 20))
    ImageDraw.Draw(canvas).text((i * (W + 8) + 4, 4), t, fill=(255, 255, 255))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/region_orig_final_labels_5x.png")
print("saved", canvas.size)
