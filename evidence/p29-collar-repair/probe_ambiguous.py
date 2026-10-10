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
hair = np.asarray(ms.hair)>0; background = labels==0; collar = np.asarray(ms.collar)>0
skin = adaptive_skin_mask(original, *skin_reference_colour(original, landmarks, face_width(landmarks)), face_width(landmarks)) if False else None
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
amb = pale_win & ~knit & ~skin
knit_win = pale_win & knit & ~skin
print("pale in window:", int(pale_win.sum()))
print("  -> knit-confirmed residual:", int(knit_win.sum()))
print("  -> skin-protected:", int((pale_win & skin).sum()))
print("  -> AMBIGUOUS (final pale, original not-knit, not skin):", int(amb.sum()))
ys,xs = np.where(amb)
if amb.any(): print("     ambiguous bbox", [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
# label classes in the ambiguous zone
vals, cnts = np.unique(labels[amb], return_counts=True)
print("     label classes:", dict(zip(vals.tolist(), cnts.tolist())))
# and classes in the whole collar window under the chin
sub = labels[win & ~background]
v2,c2 = np.unique(sub, return_counts=True)
print("window classes:", dict(zip(v2.tolist(), c2.tolist())))
# what does the raw original luma/chroma look like in the ambiguous zone
if amb.any():
    print("     orig luma median", float(np.median(ol[amb])), "chroma median", float(np.median(oc[amb])))
    print("     final luma median", float(np.median(fl[amb])), "chroma median", float(np.median(fc[amb])))
    print("     sample orig RGB", original[ys[len(ys)//2], xs[len(xs)//2]], "final RGB", final[ys[len(ys)//2], xs[len(xs)//2]])
