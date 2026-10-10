import sys, numpy as np
from pathlib import Path
from PIL import Image
REPO = Path('/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent')
sys.path.insert(0, str(REPO/'backend'))
from app.core.config import settings
from app.domains.portrait.face_lock import (FACE_OVAL, adaptive_skin_mask, config_from_settings,
    face_width, skin_reference_colour)
from app.services.native_mask_inpainting import NativeMaskInpainting
from matplotlib.path import Path as MPath
orig_p = REPO/'data/inputs/IMG_3651.jpg'
final_p= REPO/'evidence/p29-submit-run/final.png'
orig = np.asarray(Image.open(orig_p).convert('RGB'))
final= np.asarray(Image.open(final_p).convert('RGB'))
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(orig_p, orig)
fw = face_width(lm)
ref,_ = skin_reference_colour(orig, lm, fw)
cfg = config_from_settings(settings)
sk = adaptive_skin_mask(orig, ref, fw, cfg)>0
print('skin reference colour', np.asarray(ref).round(0), 'face_w', round(fw,1))
def lc(a):
    a=a.astype(float); return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2], a.max(2)-a.min(2)
lo,co = lc(orig); lf,cf = lc(final)
def rep(m,name):
    n=int(m.sum())
    if n<10: print(name,'n=',n); return
    print(f'{name}: n={n}  orig RGB {np.median(orig[m],0).round(0)} luma {np.median(lo[m]):.1f} chroma {np.median(co[m]):.1f} | final luma {np.median(lf[m]):.1f} chroma {np.median(cf[m]):.1f}')
yy,xx = np.mgrid[0:orig.shape[0],0:orig.shape[1]]
pts = np.column_stack([xx.ravel(),yy.ravel()])
oval = MPath(np.asarray(lm)[list(FACE_OVAL)]).contains_points(pts).reshape(orig.shape[:2])
rep(sk & oval, 'FACIAL skin (skin AND oval)')
win = np.zeros(orig.shape[:2],bool); win[594:753, 327:679]=True
rep(sk & ~win & (yy>560), 'skin below chin OUTSIDE window')
rep(sk & win, 'skin INSIDE window')
grd = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_guarded_pale.png').convert('L'))>0
rep(grd, 'guarded_pale band')
rel = np.asarray(Image.open(REPO/'evidence/p29-collar-repair/collar_repair_mask.png').convert('L'))>0
rep(rel & ~grd, 'release (excl carve overlap)')
print('chin', np.asarray(lm)[list([1])], 'face_w', round(fw,1))
print('landmark chin idx used CHIN; report chin y via landmarks')
