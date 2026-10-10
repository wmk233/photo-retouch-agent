import sys, inspect
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting
REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
ORIG = REPO/"data/inputs/IMG_3651.jpg"
original = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(str(ORIG), original)
fw = face_width(lm); cx, cy = (float(v) for v in lm[CHIN])
ref,_ = skin_reference_colour(original, lm, fw)
skin = adaptive_skin_mask(original, ref, fw, config_from_settings(settings))>0
n, lab, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
# the component at a face pixel (sampled between the eyes)
face_px = (int(lm[1][1]), int(lm[1][0]))
face_comp = lab[face_px]
# what components cover the band x517-570 y595-692?
band = np.zeros_like(skin); band[595:693, 517:571] = True
comps = {}
for c in np.unique(lab[band & skin]):
    if c == 0: continue
    m = lab == c
    ys,xs = np.where(m)
    comps[int(c)] = (int(m.sum()), [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())], bool(c==face_comp))
for c,(px,bbox,isf) in sorted(comps.items(), key=lambda kv:-kv[1][0])[:6]:
    print(f"comp {c} px={px} bbox={bbox} is_face_component={isf}")
print("face component id", face_comp, "px", int((lab==face_comp).sum()))
# does the face component reach down to y>640 near the neck?
m = lab==face_comp
for y in (600, 620, 640, 660, 680):
    row = np.where(m[y, 440:600])[0]
    print(f"y={y} face-comp x range:", (int(row.min())+440, int(row.max())+440) if row.size else None)
