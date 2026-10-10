"""Character maps of the collar region: original class vs final class vs unchanged-ness."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0, str(REPO / "backend"))
from app.core.config import settings
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks
from app.domains.portrait.face_lock import (
    FACE_OVAL, adaptive_skin_mask, config_from_settings, face_width, fill_polygon, skin_reference_colour)
from app.services.native_mask_inpainting import NativeMaskInpainting

orig_p = REPO / "data/inputs/IMG_3651.jpg"
orig = np.asarray(Image.open(orig_p).convert("RGB"))
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
labels = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/labels_fresh.png"))
H, W = labels.shape
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(str(orig_p), orig)
ms = build_semantic_edit_masks(orig, lm, labels)
protected = np.asarray(ms.protected) > 0
hair = np.asarray(ms.hair) > 0
background = labels == 0
fw = face_width(lm); chin_x, chin_y = (float(v) for v in lm[CHIN])
ref, _ = skin_reference_colour(orig, lm, fw)
skin = adaptive_skin_mask(orig, ref, fw, config_from_settings(settings)) > 0
oval = fill_polygon((H, W), np.asarray(lm)[list(FACE_OVAL)]) > 0
def lc(a):
    a = a.astype(np.float32)
    return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
ol, oc = lc(orig); fl, fc = lc(final)
change = np.abs(orig.astype(np.int16)-final.astype(np.int16)).max(2)

# --- the r9 residual block rule ---
o_hue = np.zeros((H,W), np.float32); o_sat = np.zeros((H,W), np.float32)
a = final.astype(np.float32)/255.0
mx, mn = a.max(2), a.min(2); d = mx-mn
sat = np.where(mx>1e-6, d/np.maximum(mx,1e-6), 0.0)
r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
hue = np.zeros((H,W), np.float32)
m = (mx==r)&(d>1e-6); hue[m] = (60*((g-b)/np.maximum(d,1e-6))%360)[m]
m = (mx==g)&(d>1e-6); hue[m] = (60*((b-r)/np.maximum(d,1e-6))+120)[m]
m = (mx==b)&(d>1e-6); hue[m] = (60*((r-g)/np.maximum(d,1e-6))+240)[m]
olive = (hue>=45)&(hue<=115)&(sat>0.12)
x0,y0,x1,y1 = 440, 550, 660, 770
band = np.zeros((H,W),bool); band[y0:y1,400:700]=True
vis = (fl>140)&(fc<55)&~olive&band
resid = (fl>132)&(fc>25)&(fc<72)&(ol>140)&(oc<50)&~skin&~hair&~background&~olive
window = np.zeros((H,W),bool)
window[int(chin_y):int(chin_y+0.62*fw), int(chin_x-0.62*fw):int(chin_x+0.75*fw)]=True
residual = ((fl>132)&(fc>25)&(fc<72)&(ol>140)&(oc<50)&~skin&window&~hair&~background&~olive)
guard_pale = ((fl>132)&(fc>25)&(fc<72)&window&~background&~hair&~olive&skin)
out = []
out.append(f"chin=({chin_x:.1f},{chin_y:.1f}) face_w={fw:.1f} labels={sorted(np.unique(labels).tolist())}")
out.append(f"window x{int(chin_x-0.62*fw)}-{int(chin_x+0.75*fw)} y{int(chin_y)}-{int(chin_y+0.62*fw)}")
out.append("")
out.append("=== ORIGINAL CLASS MAP (x440-660 step2) ===")
out.append("legend B=bg H=hair S=orig-skin-model K=knit(l>140,c<50) k=bright-neutral(l>150,c<45) .=neither")
out.append("     " + "".join(str((x//10)%10) for x in range(x0,x1,2)))
for y in range(y0,y1,2):
    row=[]
    for x in range(x0,x1,2):
        if background[y,x]: row.append('B')
        elif hair[y,x]: row.append('H')
        elif skin[y,x]: row.append('S')
        elif ol[y,x]>150 and oc[y,x]<45: row.append('k')
        elif ol[y,x]>140 and oc[y,x]<50: row.append('K')
        else: row.append('.')
    out.append(f"{y:4d} "+"".join(row))
out.append("")
out.append("=== FINAL CLASS MAP ===")
out.append("legend o=olive @=pale(132<l,25<c<72) .=neither")
out.append("     " + "".join(str((x//10)%10) for x in range(x0,x1,2)))
for y in range(y0,y1,2):
    row=[]
    for x in range(x0,x1,2):
        if olive[y,x]: row.append('o')
        elif fl[y,x]>132 and 25<fc[y,x]<72: row.append('@')
        else: row.append('.')
    out.append(f"{y:4d} "+"".join(row))
out.append("")
out.append("=== UNCHANGED MAP (max channel change < 10 => render left original pixel) ===")
out.append("     " + "".join(str((x//10)%10) for x in range(x0,x1,2)))
for y in range(y0,y1,2):
    out.append(f"{y:4d} "+"".join('#' if change[y,x]<10 else '.' for x in range(x0,x1,2)))

# stats for the r9 largest blob bbox
bx,by,bw,bh = 514,578,51,111
sl = (slice(by,by+bh), slice(bx,bx+bw))
m_skin = skin[sl]; m_knit = (ol[sl]>140)&(oc[sl]<50); m_bright=(ol[sl]>150)&(oc[sl]<45)
m_chg = change[sl]<10
out.append("")
out.append(f"=== BLOCK [514,578,51,111] ({bw*bh} box) ===")
out.append(f"orig skin-model px      {int(m_skin.sum())} / {m_skin.size} ({100*m_skin.mean():.1f}%)")
out.append(f"orig knit(l140,c50) px  {int(m_knit.sum())} ({100*m_knit.mean():.1f}%)")
out.append(f"orig bright-neutral px  {int(m_bright.sum())} ({100*m_bright.mean():.1f}%)")
out.append(f"unchanged(chg<10) px    {int(m_chg.sum())} ({100*m_chg.mean():.1f}%)")
out.append(f"median orig RGB {np.median(orig[sl].reshape(-1,3),0)} final RGB {np.median(final[sl].reshape(-1,3),0)}")
np.save(REPO/"evidence/p29-collar-residual-mask/_masks.npy", {
  "skin":skin,"hair":hair,"background":background,"oval":oval,"protected":protected,
  "olive":olive,"change":change,"ol":ol,"oc":oc,"fl":fl,"fc":fc,"residual":residual,
  "guard_pale":guard_pale,"vis":vis,"window":window}, allow_pickle=True)
print("\n".join(out))
