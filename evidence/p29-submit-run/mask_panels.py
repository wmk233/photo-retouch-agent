from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-submit-run"
PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
orig = Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")
final = Image.open(OUT / "final.png").convert("RGB")
cloth = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127

X0, Y0, X1, Y1 = 380, 540, 620, 780
S = 5
sub_c, sub_p = cloth[Y0:Y1, X0:X1], prot[Y0:Y1, X0:X1]
print("crop pixels:", sub_c.size, "clothing in crop:", int(sub_c.sum()), "protected in crop:", int(sub_p.sum()))
print("overlap in crop:", int((sub_c & sub_p).sum()))
# outside both
ob = (~sub_c) & (~sub_p)
print("outside-both in crop:", int(ob.sum()))
o = np.asarray(orig).astype(np.int16)[Y0:Y1, X0:X1]
f = np.asarray(final).astype(np.int16)[Y0:Y1, X0:X1]
d = np.abs(o - f).max(axis=2)
print("outside-both: max diff", int(d[ob].max()), "share>25", round(float((d[ob] > 25).mean()), 5),
      "share==0", round(float((d[ob] == 0).mean()), 5))
print("inside-clothing: share>25", round(float((d[sub_c] > 25).mean()), 5))
print("inside-protected: share>25", round(float((d[sub_p] > 25).mean()), 5), "max", int(d[sub_p].max()))

def overlay(im, mask, rgb, text):
    a = np.asarray(im).astype(np.float32).copy()
    sub = mask[Y0:Y1, X0:X1]
    up = np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)) > 127
    t = a[Y0:Y1, X0:X1]
    t = np.asarray(Image.fromarray(t.astype(np.uint8)).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)).astype(np.float32)
    t[up] = t[up] * 0.5 + np.array(rgb, np.float32) * 0.5
    out = Image.fromarray(t.astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width - 1, 30], fill=(0, 0, 0))
    d.text((8, 9), text, fill=(255, 255, 255))
    return out

tiles = [overlay(orig, cloth, (0, 255, 0), "ORIGINAL + clothing EDIT mask (green)"),
         overlay(orig, prot, (255, 0, 0), "ORIGINAL + protected mask (red)"),
         overlay(final, cloth, (0, 255, 0), "FINAL + clothing EDIT mask (green)"),
         overlay(final, prot, (255, 0, 0), "FINAL + protected mask (red)")]
W = tiles[0].width * 2 + 12
H = tiles[0].height * 2 + 12
canvas = Image.new("RGB", (W, H), (24, 24, 24))
for i, t in enumerate(tiles):
    canvas.paste(t, ((i % 2) * (tiles[0].width + 12), (i // 2) * (tiles[0].height + 12)))
canvas.save(OUT / "mask_panels5x.png")
print("saved", OUT / "mask_panels5x.png", canvas.size)
