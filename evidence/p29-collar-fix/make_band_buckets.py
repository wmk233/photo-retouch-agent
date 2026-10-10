"""Colour-code the 805-px band by luma bucket over the original, 8x."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting

ORIG = ROOT/"data/inputs/IMG_3651.jpg"
image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(ROOT/"evidence/p29-neck-fix/p29_labels.png"))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(after.protected) > 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
r,g,b = (image[:,:,i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)
ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])
below = np.arange(H)[:,None] >= round(chin_y)
band = prot & below & (labels == hp.FACE) & ~skin

ov = image.copy()
dark = band & (luma < 0.35*ref_luma)
mid  = band & (luma >= 0.35*ref_luma) & (luma <= 1.0*ref_luma)
bri  = band & (luma > 1.0*ref_luma)
ov[dark] = np.array([120, 0, 255], np.uint8)      # violet = deep shadow
ov[mid]  = np.array([255, 140, 0], np.uint8)      # orange = mid
ov[bri]  = np.array([0, 255, 0], np.uint8)        # green  = bright white fabric
Y0,Y1,X0,X1 = 575, 625, 380, 580
img = Image.fromarray(ov[Y0:Y1, X0:X1]).resize(((X1-X0)*9, (Y1-Y0)*9), Image.NEAREST)
img.save(ROOT/"evidence/p29-collar-fix/band_buckets_9x.png")
print("dark", int(dark.sum()), "mid", int(mid.sum()), "bright", int(bri.sum()))
print("chroma<45 among mid+bright:", int(((mid|bri) & (chroma<45)).sum()), "chroma 45-70:", int(((mid|bri)&(chroma>=45)&(chroma<70)).sum()))
print("wrote", img.size)
