import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from tools.build_collar_repair_mask import (
    _luma_chroma, _hue_sat, PALE_LUMA_MIN, PALE_CHROMA_MIN, PALE_CHROMA_MAX,
    SOURCE_KNIT_LUMA_MIN, SOURCE_KNIT_CHROMA_MAX, OLIVE_HUE_MIN, OLIVE_HUE_MAX, OLIVE_SAT_MIN,
)
REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
final = np.asarray(Image.open(REPO/"evidence/p29-submit-run/final.png").convert("RGB"))
orig = np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
fl, fc = _luma_chroma(final); ol, oc = _luma_chroma(orig)
fh, fs = _hue_sat(final)
print("hue range in center region:")
for y in range(640, 710, 20):
    for x in range(470, 590, 20):
        pf = fl[y,x]>PALE_LUMA_MIN and PALE_CHROMA_MIN<fc[y,x]<PALE_CHROMA_MAX
        sk = ol[y,x]>SOURCE_KNIT_LUMA_MIN and oc[y,x]<SOURCE_KNIT_CHROMA_MAX
        ov = OLIVE_HUE_MIN<=fh[y,x]<=OLIVE_HUE_MAX and fs[y,x]>OLIVE_SAT_MIN
        print(f"({x},{y}) hue={fh[y,x]:.0f} sat={fs[y,x]:.2f} pale={pf} knit={sk} olive={ov} RGB={final[y,x]}")
