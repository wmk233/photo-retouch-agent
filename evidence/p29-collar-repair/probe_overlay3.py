import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
import tools.build_collar_repair_mask as B
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
ORIG = REPO/"data/inputs/IMG_3651.jpg"
final = np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
original = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
h,w = labels.shape
svc = NativeMaskInpainting(settings)
landmarks = svc._detect_landmarks(str(ORIG), original)
ms = build_semantic_edit_masks(original, landmarks, labels)
hair = np.asarray(ms.hair)>0; background = labels==0
fw = face_width(landmarks); cx, cy = (float(v) for v in landmarks[CHIN])
ref,_ = skin_reference_colour(original, landmarks, fw)
skin = adaptive_skin_mask(original, ref, fw, config_from_settings(settings))>0
fl, fc = B._luma_chroma(final); ol, oc = B._luma_chroma(original); fh, fs = B._hue_sat(final)
pale = (fl>B.PALE_LUMA_MIN)&(fc>B.PALE_CHROMA_MIN)&(fc<B.PALE_CHROMA_MAX)
knit = (ol>B.SOURCE_KNIT_LUMA_MIN)&(oc<B.SOURCE_KNIT_CHROMA_MAX)
olive = (fh>=B.OLIVE_HUE_MIN)&(fh<=B.OLIVE_HUE_MAX)&(fs>B.OLIVE_SAT_MIN)
win = np.zeros((h,w),bool)
win[int(cy):int(cy+B.WINDOW_DOWN_FACE_WIDTHS*fw), max(int(cx-B.WINDOW_LEFT_FACE_WIDTHS*fw),0):min(int(cx+B.WINDOW_RIGHT_FACE_WIDTHS*fw),w)] = True
pale_win = pale & win & ~background & ~olive & ~hair
residual = pale_win & knit & ~skin
skin_pale = pale_win & skin
amb = pale_win & ~knit & ~skin
rel = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>0
img = final.copy()
img[skin_pale] = [0,90,255]
img[amb] = [255,220,0]
img[residual & ~rel] = [0,200,0]
img[rel] = [255,0,0]
crop = img[580:740,430:660]
zc = np.asarray(Image.fromarray(crop).resize((crop.shape[1]*4, crop.shape[0]*4), Image.NEAREST))
Image.fromarray(zc).save(REPO/"evidence/p29-collar-repair/probe_categories_4x.png")
print("red=release blue=skin-protected-pale green=knit-pale-not-released yellow=ambiguous")
print("residual", int(residual.sum()), "released", int(rel.sum()), "skin_pale", int(skin_pale.sum()), "amb", int(amb.sum()))
