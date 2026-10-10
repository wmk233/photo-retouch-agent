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
labels = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/labels_fresh.png"))
Y0,Y1,X0,X1 = 580,740,430,650
# residual mask from the tool (already on disk) plus recomputed residual
res = np.asarray(Image.open(REPO/"evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>0
orig = np.asarray(Image.open(ORIG).convert("RGB"))
fig = final.copy()
# outline residual in blue (from tool output)
import cv2
cnt,_ = cv2.findContours((res*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(fig, cnt, -1, (0,0,255), 1)
def z(img):
    c = img[Y0:Y1, X0:X1]
    return np.asarray(Image.fromarray(c).resize((c.shape[1]*4, c.shape[0]*4), Image.NEAREST))
row = np.concatenate([z(orig), z(final), z(fig)], axis=1)
Image.fromarray(row).save(REPO/"evidence/p29-collar-repair/probe_map_panel.png")
print(row.shape)
# ascii residual in region
print("   " + "".join(f"{x%100:02d}"[-2:] for x in range(440,620,10)))
for y in range(590,720,10):
    line = "".join("#" if res[y,x] else "." for x in range(440,620,10))
    print(f"{y} {line}")
