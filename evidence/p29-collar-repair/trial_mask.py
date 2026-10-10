"""Trial collar_repair_mask: nearest rule set, visualised for review."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.domains.portrait import human_parsing as hp
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import adaptive_skin_mask, config_from_settings, face_width, skin_reference_colour
from app.services.native_mask_inpainting import NativeMaskInpainting

OUT = ROOT / "evidence/p29-collar-repair"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
FINAL = ROOT / "evidence/p29-submit-run/final.png"
LABELS = OUT / "labels_fresh.png"

orig = np.asarray(Image.open(ORIG).convert("RGB"))
final = np.asarray(Image.open(FINAL).convert("RGB"))
labels = np.asarray(Image.open(LABELS))
H, W = labels.shape
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, orig)
ms = build_semantic_edit_masks(orig, lm, labels)
prot = np.asarray(ms.protected) > 0
hair = np.asarray(ms.hair) > 0
cloth = np.asarray(ms.clothing) > 0
collar = np.asarray(ms.collar) > 0
bg = labels == 0
face_w = face_width(lm)
chin = lm[CHIN]; chin_y, chin_x = float(chin[1]), float(chin[0])
reference, _ = skin_reference_colour(orig, lm, face_w)
lock_cfg = config_from_settings(settings)
skin_final = adaptive_skin_mask(final, reference, face_w, lock_cfg) > 0

def luma_chroma(img):
    a = img.astype(np.float32)
    return (0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2))

def hsv(img):
    a = img.astype(np.float32)/255.0
    mx=a.max(2); mn=a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0)
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    hue=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; hue[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; hue[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; hue[i]=60*((r[i]-g[i])/d[i])+240
    return hue, np.where(m, sat, 0), mx

ol, oc = luma_chroma(orig)
fl, fc = luma_chroma(final)
hue, sat, val = hsv(final)

orig_knit = (ol > 140) & (oc < 50)
final_bright = fl > 135
olive = (hue >= 40) & (hue <= 115) & (sat > 0.15)

window = np.zeros((H, W), bool)
window[int(chin_y):int(chin_y + 0.65*face_w), int(chin_x - 1.0*face_w):int(chin_x + 1.0*face_w)] = True

core = orig_knit & final_bright & ~skin_final & ~olive
cand = core & window & ~hair & ~bg
repair = cand & ~prot
conflict = cand & prot

# tidy: drop speckle, keep components >= 300 px, close 1 px
repair_u8 = (repair*255).astype(np.uint8)
repair_u8 = cv2.morphologyEx(repair_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(repair_u8, 8)
keep = np.zeros_like(repair_u8)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 400:
        keep[lab == i] = 255
repair_final = keep > 0

n2, lab2, st2, _ = cv2.connectedComponentsWithStats((conflict*255).astype(np.uint8), 8)
report = {
 "face_w": round(face_w,1), "chin": [round(chin_x,1), round(chin_y,1)],
 "orig_knit_px": int(orig_knit.sum()),
 "orig_knit_in_window": int((orig_knit & window).sum()),
 "core_px": int(core.sum()), "window_px": int(window.sum()),
 "candidate_px": int(cand.sum()),
 "repair_px": int(repair_final.sum()),
 "repair_components>=400": int((st2.shape[0]-0) if False else (keep>0).sum()),
 "conflict_protected_px": int(conflict.sum()),
 "core_breakdown": {
   "skin_final": int((orig_knit & final_bright & skin_final & window & ~hair & ~bg).sum()),
   "olive_final": int((orig_knit & final_bright & olive & window & ~hair & ~bg).sum()),
   "not_bright": int((orig_knit & ~final_bright & window & ~hair & ~bg).sum()),
   "hair_or_bg": int((orig_knit & final_bright & (hair|bg)).sum()),
 },
}
big = [(int(st2[i,cv2.CC_STAT_AREA]), [int(st2[i,0]),int(st2[i,1]),int(st2[i,2]),int(st2[i,3])]) for i in range(1, st2.shape[0]) if st2[i,cv2.CC_STAT_AREA] >= 100]
report["conflict_blobs>=100"] = sorted(big, reverse=True)[:10]
print(json.dumps(report, indent=1))
json.dump(report, open(OUT/"trial_mask.json","w"), indent=1)

vis = final.copy()
vis[repair_final] = (vis[repair_final]*0.25 + np.array([255,0,0])*0.75).astype(np.uint8)
vis[conflict] = (vis[conflict]*0.25 + np.array([255,255,0])*0.75).astype(np.uint8)
Image.fromarray(vis).save(OUT/"trial_mask_overlay.png")
Image.fromarray(keep).save(OUT/"trial_mask_gray.png")
crop = Image.fromarray(vis).crop((290,540,700,850)).resize((820,620), Image.NEAREST)
crop.save(OUT/"trial_mask_overlay_zoom2x.png")
print("overlay written")
