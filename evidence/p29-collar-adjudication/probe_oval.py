from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.domains.portrait.face_lock import FACE_OVAL, fill_polygon
from app.services.native_mask_inpainting import NativeMaskInpainting
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
svc=NativeMaskInpainting(settings); lm=np.asarray(svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original))
h,w=original.shape[:2]
oval=fill_polygon((h,w), np.asarray(lm)[list(FACE_OVAL)])>0
block=np.zeros((h,w),bool); block[578:594,523:559]=True
m=block&oval
print("block px:",int(block.sum())," inside face-oval:",int(m.sum())," outside:",int((block&~oval).sum()))
# distance of block pixels to the oval boundary
inv=(~oval).astype(np.uint8)
d_in=cv2.distanceTransform((~block).astype(np.uint8),cv2.DIST_L2,5)
# distance from each block px to nearest non-oval px
d_out=cv2.distanceTransform((~oval).astype(np.uint8),cv2.DIST_L2,5)
print("per-px dist to oval edge: inside-block median",round(float(np.median(d_out[block])),1))
print("oval bbox:",[int(v) for v in (np.where(oval.any(0))[0].min(),np.where(oval.any(1))[0].min(),
      np.where(oval.any(0))[0].max(),np.where(oval.any(1))[0].max())])
# jaw contour row sample
print("face-oval rightmost x per row (y=570..600):")
for y in range(570,601,2):
    xs=np.where(oval[y])[0]
    print(f"   y={y}: oval x in [{xs.min()},{xs.max()}]" if xs.size else f"   y={y}: none")
