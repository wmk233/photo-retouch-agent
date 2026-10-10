"""Distribution of the 805-px band, and the same region in the two renders."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(ROOT / "evidence/p29-neck-fix/p29_labels.png"))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(after.protected) > 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
band = prot & (np.zeros((H,W),bool) | (np.arange(H)[:,None] >= round(chin_y))) & (labels == hp.FACE) & ~skin

def stats(img, m):
    if not m.any(): return None
    px = np.asarray(img)[m].astype(np.float32)
    l = 0.299*px[:,0]+0.587*px[:,1]+0.114*px[:,2]
    ch = px.max(1)-px.min(1)
    return {"n": int(m.sum()), "luma_mean": round(float(l.mean()),1), "luma_med": round(float(np.median(l)),1),
            "chroma_mean": round(float(ch.mean()),1), "chroma_med": round(float(np.median(ch)),1),
            "rgb_mean": [round(float(v),1) for v in px.mean(0)]}

ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])
out = {"reference_luma": round(ref_luma,1), "reference_rgb": [round(float(v),1) for v in reference]}
out["orig_band"] = stats(Image.open(ORIG), band)
out["final_band"] = stats(Image.open(ROOT/"evidence/p29-submit-run/final.png"), band)
out["clothpass_band"] = stats(Image.open(ROOT/"evidence/p29-submit-run/clothing_pass.png"), band)

# reference skin for contrast: pixels just above the chin inside the face oval
r,g,b = (image[:,:,i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)
skin_band = prot & skin & (np.arange(H)[:,None] >= round(chin_y-40)) & (np.arange(H)[:,None] < round(chin_y))
out["skin_reference_region"] = stats(Image.open(ORIG), skin_band)
# hist of the band by luma ratio
ratio = luma[band]/ref_luma
hist, edges = np.histogram(ratio, bins=[0,0.2,0.3,0.35,0.4,0.5,0.6,0.8,1.0,1.5])
out["band_luma_ratio_hist"] = {f"{edges[i]:.2f}-{edges[i+1]:.2f}": int(hist[i]) for i in range(len(hist))}
chr_hist, ced = np.histogram(chroma[band], bins=[0,20,30,40,45,55,70,100,200])
out["band_chroma_hist"] = {f"{ced[i]:.0f}-{ced[i+1]:.0f}": int(chr_hist[i]) for i in range(len(chr_hist))}
print(json.dumps(out, indent=2))
json.dump(out, open(ROOT/"evidence/p29-collar-fix/collar_band_profile.json","w"), indent=2)
