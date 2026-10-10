from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-submit-run"
PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
orig = Image.open(ROOT / "data/inputs/IMG_3651.jpg").convert("RGB")
final = Image.open(OUT / "final.png").convert("RGB")
fail = Image.open(ROOT / "artifacts/p29-fix-run/result_0.png").convert("RGB")
cloth = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127

X0, Y0, X1, Y1 = 380, 540, 620, 780
S = 6

def crop(im):
    return im.crop((X0, Y0, X1, Y1)).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)

def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 30], fill=(0, 0, 0))
    d.text((8, 9), text, fill=(255, 255, 255))
    return im

# mask overlay on final
ov = np.asarray(final).copy()
sub_c = cloth[Y0:Y1, X0:X1]
sub_p = prot[Y0:Y1, X0:X1]
edge_c = sub_c & ~np.pad(sub_c, 1)[:-2, 1:-1]
ov_c = Image.fromarray(ov[Y0:Y1, X0:X1]).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)
arr = np.asarray(ov_c).copy()
up_c = np.asarray(Image.fromarray(sub_c.astype(np.uint8) * 255).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)) > 127
up_p = np.asarray(Image.fromarray(sub_p.astype(np.uint8) * 255).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST)) > 127
ec = up_c & ~np.pad(up_c, 1)[:-2, 1:-1]
ep = up_p & ~np.pad(up_p, 1)[:-2, 1:-1]
arr[ec] = [255, 0, 255]
arr[ep] = [0, 255, 255]
overlay = label(Image.fromarray(arr), "FINAL + clothing mask (magenta) / protected (cyan)")

# diff vs original
o = np.asarray(orig.convert("RGB")).astype(np.int16)
f = np.asarray(final.convert("RGB")).astype(np.int16)
d = np.abs(o - f).max(axis=2)
vis = np.asarray(final).copy()
vis[d > 25] = [255, 40, 40]
diff = label(Image.fromarray(vis[Y0:Y1, X0:X1]).resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.NEAREST), "FINAL, pixels changed >25 vs original = red")

tiles = [label(crop(orig), "ORIGINAL"), label(crop(final), "FINAL (this run)"),
         overlay, diff]
W = tiles[0].width * 2 + 12
H = tiles[0].height * 2 + 12
canvas = Image.new("RGB", (W, H), (24, 24, 24))
canvas.paste(tiles[0], (0, 0))
canvas.paste(tiles[1], (tiles[0].width + 12, 0))
canvas.paste(tiles[2], (0, tiles[0].height + 12))
canvas.paste(tiles[3], (tiles[0].width + 12, tiles[0].height + 12))
canvas.save(OUT / "throat_diag6x.png")
print("saved", OUT / "throat_diag6x.png", canvas.size)
# extra: same for the previous failure
canvas2 = Image.new("RGB", (tiles[0].width * 2 + 12, tiles[0].height), (24, 24, 24))
canvas2.paste(label(crop(fail), "LAST FAILED (pale ghost)"), (0, 0))
canvas2.paste(label(crop(final), "FINAL (this run)"), (tiles[0].width + 12, 0))
canvas2.save(OUT / "throat_fail_vs_final6x.png")
print("saved", OUT / "throat_fail_vs_final6x.png", canvas2.size)
