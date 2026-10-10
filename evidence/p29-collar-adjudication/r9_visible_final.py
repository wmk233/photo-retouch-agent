"""Round-9 independent check: is the visible pale cloth in the FINAL covered by the edit region?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))

from tools.build_collar_repair_mask import (  # noqa: E402
    _luma_chroma, _hue_sat, _blob_list,
)
import cv2  # noqa: E402

EVID = REPO / "evidence/p29-collar-repair"
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
release = np.asarray(Image.open(EVID / "collar_repair_mask.png")) > 0
carve = np.asarray(Image.open(EVID / "collar_repair_carve.png")) > 0
guarded = np.asarray(Image.open(EVID / "collar_repair_guarded_pale.png")) > 0
protected_skin = np.asarray(Image.open(EVID / "collar_repair_protected_skin.png")) > 0
labels = np.asarray(Image.open(EVID / "labels_fresh.png"))

edit = release | carve
ol, oc = _luma_chroma(orig)
fl, fc = _luma_chroma(final)
fh, fs = _hue_sat(final)

# visible pale cloth in the FINAL: bright, low/mid chroma, not the olive blouse hue
olive = (fh >= 45) & (fh <= 115) & (fs > 0.12)
pale = (fl > 140) & (fc < 55) & ~olive

band = np.zeros_like(pale)
band[560:760, 400:640] = True   # neck + collar area only
pale &= band

print("=== FINAL visible pale cloth inside neck band ===")
print("pale_px", int(pale.sum()))
for name, m in (("release", release), ("carve", carve), ("guarded", guarded),
                ("edit(rel|carve)", edit)):
    print(f"  in {name:16s}: {int((pale & m).sum())}")
uncovered = pale & ~edit
print("  NOT in edit region:", int(uncovered.sum()))

print("\n=== uncovered blobs (>=40 px) ===")
for b in sorted(_blob_list(uncovered), key=lambda d: -d["px"])[:12]:
    x, y, w, h = b["bbox"]
    sel = np.zeros_like(uncovered); sel[y:y+h, x:x+w] = True; sel &= uncovered
    omed = np.median(orig[sel], 0).astype(int)
    fmed = np.median(final[sel], 0).astype(int)
    lab = np.bincount(labels[sel].ravel(), minlength=19)
    top = np.argsort(-lab)[:3]
    print(f"  px={b['px']:5d} bbox={b['bbox']} orig={omed} final={fmed} "
          f"labels={[(int(t), int(lab[t])) for t in top if lab[t] > 0]}")
