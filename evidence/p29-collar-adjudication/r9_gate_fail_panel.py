"""Round-9 decisive panel: the residual white cloth in the FINAL lies outside the edit region."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))
from tools.build_collar_repair_mask import _luma_chroma  # noqa: E402

EVID = REPO / "evidence/p29-collar-repair"
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
release = np.asarray(Image.open(EVID / "collar_repair_mask.png")) > 0
carve = np.asarray(Image.open(EVID / "collar_repair_carve.png")) > 0
guarded = np.asarray(Image.open(EVID / "collar_repair_guarded_pale.png")) > 0
prot_skin = np.asarray(Image.open(EVID / "collar_repair_protected_skin.png")) > 0

fl, fc = _luma_chroma(final)
mx = final.max(2).astype(np.float32); mn = final.min(2).astype(np.float32)
ch = np.where(mx - mn == 0, 1.0, mx - mn)
r, g, b = [final[:, :, i].astype(np.float32) for i in range(3)]
hue = np.zeros_like(ch)
for c, val in ((r, (60 * ((g - b) / ch) % 360)), (g, 60 * ((b - r) / ch) + 120), (b, 60 * ((r - g) / ch) + 240)):
    hue[mx == c] = val[mx == c]
sat = ch / np.maximum(mx, 1)
band = np.zeros_like(release); band[560:760, 400:640] = True
resid = (fl > 140) & (fc < 55) & band & ~(release | carve) & ~((hue >= 45) & (hue <= 115) & (sat > 0.12))

S = 5
x0, y0, x1, y1 = 430, 560, 610, 740

def crop(a):
    return a[y0:y1, x0:x1]

def up(a):
    return np.asarray(Image.fromarray(a).resize(((x1 - x0) * S, (y1 - y0) * S), Image.NEAREST))

def lab(img, text, color=(255, 255, 0)):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 300, 26], fill=(0, 0, 0)); d.text((6, 8), text, fill=color); return im

panels = []
panels.append(lab(up(crop(orig)), "ORIGINAL 5x"))
panels.append(lab(up(crop(final)), "FINAL 5x"))
fin = up(crop(final)).copy()
grd = up(crop(guarded))
rel = up(crop(release | carve))
skn = up(crop(prot_skin))
fin[grd] = (fin[grd] * 0.45 + np.array([255, 40, 40]) * 0.55).astype(np.uint8)
fin[rel] = (fin[rel] * 0.45 + np.array([0, 210, 255]) * 0.55).astype(np.uint8)
fin_im = lab(fin, "FINAL + guarded(red) release(cyan)", (255, 255, 255))
panels.append(fin_im)
fin2 = up(crop(final)).copy()
rs = np.zeros_like(resid); rs[y0:y1, x0:x1] = resid[y0:y1, x0:x1]
rsu = up(rs)
fin2 = (fin2 * 0.35).astype(np.uint8)
fin2[rsu] = np.array([0, 255, 0], np.uint8)
fin2[rel] = (fin2[rel] * 0.5 + np.array([0, 90, 160]) * 0.5).astype(np.uint8)
panels.append(lab(fin2, f"uncovered pale residue = GREEN ({int(resid.sum())} px)", (255, 255, 255)))

W, H = panels[0].size
out = Image.new("RGB", (W * 2, H * 2), (20, 20, 20))
for i, p in enumerate(panels):
    out.paste(p, ((i % 2) * W, (i // 2) * H))
dest = REPO / "evidence/p29-collar-adjudication/r9_gate_fail_panel.png"
out.save(dest)
print("saved", dest, "resid_px", int(resid.sum()))
