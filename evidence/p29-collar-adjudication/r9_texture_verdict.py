"""Round-9: texture evidence for the residual tube (fabric vs skin) on the ORIGINAL."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))
from tools.build_collar_repair_mask import _luma_chroma, _blob_list  # noqa: E402

EVID = REPO / "evidence/p29-collar-repair"
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
release = np.asarray(Image.open(EVID / "collar_repair_mask.png")) > 0
carve = np.asarray(Image.open(EVID / "collar_repair_carve.png")) > 0
guarded = np.asarray(Image.open(EVID / "collar_repair_guarded_pale.png")) > 0
prot_skin = np.asarray(Image.open(EVID / "collar_repair_protected_skin.png")) > 0

ol, oc = _luma_chroma(orig)
fl, fc = _luma_chroma(final)
olive = False
# high-frequency energy of the ORIGINAL luma
lap = cv2.Laplacian(ol.astype(np.float32), cv2.CV_32F)
hf = cv2.GaussianBlur(np.abs(lap), (0, 0), 3.0)

band = np.zeros_like(release)
band[560:760, 400:640] = True
uncovered = (fl > 140) & (fc < 55) & band & ~(release | carve)
# drop the olive-hued blouse (hue test on final)
h_, s_ = None, None
mx = final.max(2).astype(np.float32); mn = final.min(2).astype(np.float32)
ch = np.where(mx - mn == 0, 1.0, mx - mn)
r, g, b = [final[:, :, i].astype(np.float32) for i in range(3)]
hue = np.zeros_like(ch)
m = (mx == r); hue[m] = (60 * ((g - b) / ch) % 360)[m]
m = (mx == g); hue[m] = (60 * ((b - r) / ch) + 120)[m]
m = (mx == b); hue[m] = (60 * ((r - g) / ch) + 240)[m]
sat = np.where(mx == 0, 0.0, ch / np.maximum(mx, 1))
uncovered &= ~((hue >= 45) & (hue <= 115) & (sat > 0.12))

blobs = sorted(_blob_list(uncovered), key=lambda d: -d["px"])

def stats(mask, name):
    if mask.sum() == 0:
        print(f"{name:34s} n=0"); return
    print(f"{name:34s} n={int(mask.sum()):6d} orig_luma_med={np.median(ol[mask]):6.1f} "
          f"orig_chroma_med={np.median(oc[mask]):5.1f} HF_med={np.median(hf[mask]):5.2f} "
          f"HF_p75={np.percentile(hf[mask],75):5.2f}")

print("=== ROUND-9 texture verdict (ORIGINAL luma, 5x5-ish |Laplacian|) ===")
for b in blobs[:4]:
    x, y, w, h = b["bbox"]
    sel = np.zeros_like(uncovered); sel[y:y+h, x:x+w] = True; sel &= uncovered
    stats(sel, f"uncovered blob {b['px']}px {b['bbox']}")

stats(release & band, "release (adjudicated fabric)")
stats(carve, "carve (adjudicated fabric)")
stats(guarded & band, "guarded (held back)")
stats(prot_skin & band, "protected real skin")
# a pure-skin reference patch on the throat/cheek
ref = np.zeros_like(release); ref[596:640, 470:520] = True; ref &= prot_skin
stats(ref, "throat skin ref 470:520,596:640")
