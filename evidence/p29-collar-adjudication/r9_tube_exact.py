from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))
from tools.build_collar_repair_mask import _luma_chroma, _hue_sat, _blob_list  # noqa: E402

EVID = REPO / "evidence/p29-collar-repair"
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
release = np.asarray(Image.open(EVID / "collar_repair_mask.png")) > 0
carve = np.asarray(Image.open(EVID / "collar_repair_carve.png")) > 0
guarded = np.asarray(Image.open(EVID / "collar_repair_guarded_pale.png")) > 0
prot = np.asarray(Image.open(EVID / "collar_repair_protected_skin.png")) > 0
labels = np.asarray(Image.open(EVID / "labels_fresh.png"))

x, y, w, h = 514, 578, 51, 111
S = 8
print(f"=== tube bbox {[x,y,w,h]} breakdown (px) ===")
sl = (slice(y, y+h), slice(x, x+w))
for n, m in (("release", release), ("carve", carve), ("guarded", guarded), ("protected_skin", prot)):
    print(f"  {n:15s}: {int(m[sl].sum())}")
fl, fc = _luma_chroma(final); ol, oc = _luma_chroma(orig)
olive = (lambda hs: (hs[0] >= 45) & (hs[0] <= 115) & (hs[1] > 0.12))(_hue_sat(final))
pale = (fl > 140) & (fc < 55) & ~olive
print(f"  final pale (>140,<55,not olive): {int(pale[sl].sum())}")
print(f"  of which NOT in edit: {int((pale & ~(release|carve))[sl].sum())}")
lab = np.bincount(labels[sl].ravel(), minlength=19)
print("  SCHP labels:", [(int(i), int(lab[i])) for i in np.argsort(-lab)[:4] if lab[i] > 0])

# per-row profile: what is the final colour across the tube?
for yy in range(y+4, y+h-4, 12):
    row = final[yy, x:x+w]
    print(f"  y={yy:4d} finalRGB med={np.median(row,0).astype(int)} orig med={np.median(orig[yy,x:x+w],0).astype(int)} "
          f"rel={int(release[yy,x:x+w].sum())} grd={int(guarded[yy,x:x+w].sum())}")

crop = lambda a: a[y:y+h, x:x+w]
def up(a): return np.asarray(Image.fromarray(a).resize((w*S, h*S), Image.NEAREST))
def lab_(img, t):
    im = Image.fromarray(img); ImageDraw.Draw(im).rectangle([0,0,len(t)*7+8,20], fill=(0,0,0)); ImageDraw.Draw(im).text((4,5), t, fill=(255,255,0)); return im

o = up(crop(orig)); f = up(crop(final))
ov = up(crop(final)).copy()
ov[up(crop(guarded))] = (ov[up(crop(guarded))]*0.4 + np.array([255,30,30])*0.6).astype(np.uint8)
ov[up(crop(release|carve))] = (ov[up(crop(release|carve))]*0.4 + np.array([0,220,255])*0.6).astype(np.uint8)
skm = up(crop(prot)) > 0
sk = np.where(skm[..., None], np.array([0,255,0],np.uint8), np.array([25,25,25],np.uint8)).astype(np.uint8)
panels = [lab_(o,"ORIGINAL 8x"), lab_(f,"FINAL 8x"), lab_(ov,"FINAL: guarded=RED release=CYAN"), lab_(sk,"protected_skin=GREEN")]
W,H = panels[0].size if hasattr(panels[0],'size') else (w*S,h*S)
W, H = panels[0].size
out = Image.new("RGB",(W*4,H),(20,20,20))
for i,p in enumerate(panels): out.paste(p,(i*W,0))
out.save(REPO/"evidence/p29-collar-adjudication/r9_tube_exact_8x.png")
print("saved r9_tube_exact_8x.png")
