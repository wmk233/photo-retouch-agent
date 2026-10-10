from __future__ import annotations
import numpy as np, cv2, json, sys
from pathlib import Path
from PIL import Image

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))
from app.core.config import settings
from app.domains.portrait.face_lock import (
    adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour, FACE_OVAL,
)
from app.domains.portrait.edit_mask import fill_polygon, erode_mask
from app.domains.portrait.edit_mask import CHIN
from app.services.native_mask_inpainting import NativeMaskInpainting

orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
rel = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_mask.png")) > 127
gp  = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_guarded_pale.png")) > 127

svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(REPO / "data/inputs/IMG_3651.jpg", orig)
fw = face_width(lm)
chin = lm[CHIN]
print("face_width", round(fw,2), "chin", [round(float(v),1) for v in chin])

def lc(rgb):
    a = rgb.astype(np.float32)
    luma = 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]
    chroma = a.max(2)-a.min(2)
    return luma, chroma

def tex(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap2 = lap*lap
    return np.sqrt(cv2.GaussianBlur(lap2, (0,0), 2.5))

ol, oc = lc(orig); fl, fc = lc(final)
ot = tex(orig)

# --- references ---
# skin reference patch: eroded face oval minus features (same as tool)
oval = fill_polygon(orig.shape[:2], lm[list(FACE_OVAL)])
skin_sample = erode_mask(oval, 0.30*fw)
skin_ref_px = skin_sample > 0
# knit reference: the release band (proven pale knit in original)
knit_ref_px = rel

def stats(mask, name):
    if mask.sum() == 0:
        print(f"{name}: EMPTY"); return
    yl, yc, yt = ol[mask].mean(), oc[mask].mean(), ot[mask].mean()
    d = np.abs(orig.astype(np.float32)-final.astype(np.float32)).mean(axis=2)[mask].mean()
    print(f"{name:28s} n={int(mask.sum()):6d}  orig_luma={yl:6.1f} chroma={yc:6.1f} "
          f"tex={yt:6.1f}  |orig-final|={d:6.2f}")

stats(skin_ref_px, "SKIN_ref (face oval)")
stats(knit_ref_px, "KNIT_ref (release band)")
stats(gp, "GUARDED band (3007)")
stats(gp & (ol > 140) & (oc < 50), "  guarded & orig-pale-lowchroma")
stats(gp & ~((ol > 140) & (oc < 50)), "  guarded & NOT pale-lowchroma")

# texture percentile for skin ref and knit ref
print("\ntexture percentiles (orig):")
for nm, m in [("skin", skin_ref_px), ("knit", knit_ref_px), ("guarded", gp)]:
    v = ot[m]
    print(f"  {nm:8s} p10={np.percentile(v,10):5.1f} p50={np.percentile(v,50):5.1f} p90={np.percentile(v,90):5.1f}")

# throat skin just under chin (below chin y, above band) - probe a few rows
cy = int(chin[1]); cx = int(chin[0])
print("\nchin:", cx, cy)

# where is guarded relative to chin? print bbox + centroid
ys, xs = np.where(gp)
print("guarded centroid", int(xs.mean()), int(ys.mean()), "bbox", xs.min(), ys.min(), xs.max(), ys.max())

# Save a texture map + relative-to-reference classification for visual check
np.save(REPO/"evidence/p29-collar-adjudication/_ot.npy", ot)
