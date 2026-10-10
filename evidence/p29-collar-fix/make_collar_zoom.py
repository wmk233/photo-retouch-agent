"""8x zoom of the chin/collar band: original, final, clothing pass, plus masks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
FIN = ROOT / "evidence/p29-submit-run/final.png"
CLO = ROOT / "evidence/p29-submit-run/clothing_pass.png"
OUT = ROOT / "evidence/p29-collar-fix"

Y0, Y1, X0, X1 = 570, 640, 360, 600
SCALE = 8

def crop(path):
    im = Image.open(path).convert("RGB")
    return im.crop((X0, Y0, X1, Y1)).resize(((X1 - X0) * SCALE, (Y1 - Y0) * SCALE), Image.LANCZOS)

tiles = [("ORIGINAL", crop(ORIG)), ("FINAL", crop(FIN)), ("CLOTHING PASS", crop(CLO))]
w, h = tiles[0][1].size
canvas = Image.new("RGB", (w, h * len(tiles) + 30 * len(tiles)), (20, 20, 20))
d = ImageDraw.Draw(canvas)
for i, (name, t) in enumerate(tiles):
    y = i * (h + 30)
    d.text((10, y + 8), name, fill=(255, 255, 255))
    canvas.paste(t, (0, y + 30))
canvas.save(OUT / "collar_band_orig_final_cloth_8x.png")
print("wrote", OUT / "collar_band_orig_final_cloth_8x.png", canvas.size)

# mask overlay on the original: carved collar green, recovered candidate red, protected pale yellow
import sys
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
import cv2

image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(ROOT / "evidence/p29-neck-fix/p29_labels.png"))
H, W = labels.shape
lm = NativeMaskInpainting(settings)._detect_landmarks(ORIG, image)
after = build_semantic_edit_masks(image, lm, labels)
prot = np.asarray(after.protected) > 0
collar = np.asarray(after.collar) > 0
face_w = face_width(lm)
reference, _ = skin_reference_colour(image, lm, face_w)
skin = adaptive_skin_mask(image, reference, face_w, config_from_settings(settings)) > 0
chin_y = float(lm[CHIN][1])
r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
luma = 0.299*r+0.587*g+0.114*b
chroma = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
ref_luma = float(0.299*reference[0]+0.587*reference[1]+0.114*reference[2])
below = np.zeros((H, W), bool); below[int(round(chin_y)):, :] = True
cand = prot & below & (labels == hp.FACE) & ~skin

ov = image.copy()
ov[collar] = (0.5 * ov[collar] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
ov[cand] = (0.35 * ov[cand] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
pale = cand & (luma > 0.9 * ref_luma) & (chroma < 45)
ov[pale] = np.array([255, 255, 0], np.uint8)
ovi = Image.fromarray(ov).crop((X0, Y0, X1, Y1)).resize(((X1-X0)*SCALE, (Y1-Y0)*SCALE), Image.NEAREST)
ovi.save(OUT / "collar_band_mask_overlay_8x.png")
print("wrote", OUT / "collar_band_mask_overlay_8x.png", ovi.size)
print("cand", int(cand.sum()), "pale_in_cand", int(pale.sum()), "collar", int(collar.sum()))
