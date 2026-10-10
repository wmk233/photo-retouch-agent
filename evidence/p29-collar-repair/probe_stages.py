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
print("labels shape/mode", labels.shape, Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png").mode)
h, w = labels.shape
svc = NativeMaskInpainting(settings)
landmarks = svc._detect_landmarks(str(ORIG), original)
ms = build_semantic_edit_masks(original, landmarks, labels)
protected = np.asarray(ms.protected) > 0
hair = np.asarray(ms.hair) > 0
clothing = np.asarray(ms.clothing) > 0
collar = np.asarray(ms.collar) > 0
background = labels == 0
fw = face_width(landmarks); cx, cy = (float(v) for v in landmarks[CHIN])
ref,_ = skin_reference_colour(original, landmarks, fw)
skin = adaptive_skin_mask(original, ref, fw, config_from_settings(settings)) > 0
fl, fc = B._luma_chroma(final); ol, oc = B._luma_chroma(original); fh, fs = B._hue_sat(final)
pale = (fl>B.PALE_LUMA_MIN)&(fc>B.PALE_CHROMA_MIN)&(fc<B.PALE_CHROMA_MAX)
knit = (ol>B.SOURCE_KNIT_LUMA_MIN)&(oc<B.SOURCE_KNIT_CHROMA_MAX)
olive = (fh>=B.OLIVE_HUE_MIN)&(fh<=B.OLIVE_HUE_MAX)&(fs>B.OLIVE_SAT_MIN)
win = np.zeros((h,w),bool)
win[int(cy):int(cy+B.WINDOW_DOWN_FACE_WIDTHS*fw), max(int(cx-B.WINDOW_LEFT_FACE_WIDTHS*fw),0):min(int(cx+B.WINDOW_RIGHT_FACE_WIDTHS*fw),w)] = True
print("chin", round(cx,1), round(cy,1), "face_w", round(fw,1), "window", int(win.sum()))
residual = pale & knit & ~skin & win & ~hair & ~background & ~olive
for name, m in [("pale",pale),("knit",knit),("olive",olive),("skin(all)",skin),("protected",protected),("residual",residual)]:
    print(f"{name}: total={int(m.sum())} in_collar={int((m&collar).sum())} in_clothing={int((m&clothing).sum())}")
w2 = win & ~background & ~olive
print("\nwin(non-bg,non-olive) pale&knit:", int((pale&knit&w2).sum()))
print("  -skin:", int((pale&knit&~skin&w2).sum()))
print("  -hair:", int((pale&knit&~skin&~hair&w2).sum()))
print("  -protected:", int((pale&knit&~skin&~hair&~protected&w2).sum()))
print("\nskin&win px:", int((skin&win).sum()), " knit&win px:", int((knit&win).sum()))
