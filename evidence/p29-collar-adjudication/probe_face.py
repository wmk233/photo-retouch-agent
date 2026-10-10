from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
REPO=Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
sys.path.insert(0,str(REPO/"backend"))
from app.core.config import settings
from app.services.native_mask_inpainting import NativeMaskInpainting
original=np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
svc=NativeMaskInpainting(settings); lm=svc._detect_landmarks(str(REPO/"data/inputs/IMG_3651.jpg"),original)
lm=np.asarray(lm)
print("n landmarks:",lm.shape)
names=["MOUTH_LEFT","MOUTH_RIGHT","MOUTH_TOP","MOUTH_BOTTOM","NOSE_TIP","CHIN","LEFT_EYE","RIGHT_EYE"]
for i,p in enumerate(lm):
    print(f"  #{i:2d} ({p[0]:7.1f},{p[1]:7.1f})")
