"""10x look at the chin-margin strip on the ORIGINAL, unannotated + parsed."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.domains.portrait import human_parsing as hp

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-collar-fix"

image = np.asarray(Image.open(ORIG).convert("RGB"))
labels = np.asarray(Image.open(LABELS))

Y0, Y1, X0, X1 = 586, 618, 388, 572
Z = 10
crop = image[Y0:Y1, X0:X1]
Image.fromarray(crop).resize(((X1 - X0) * Z, (Y1 - Y0) * Z), Image.NEAREST).save(
    OUT / "strip_orig_10x.png"
)

PAL = {
    0: (40, 40, 40), 2: (120, 40, 160), 4: (40, 90, 220),
    11: (230, 150, 120), 17: (30, 200, 190),
}
tint = crop.copy()
for cls, col in PAL.items():
    sel = labels[Y0:Y1, X0:X1] == cls
    tint[sel] = (0.35 * tint[sel] + 0.65 * np.array(col)).astype(np.uint8)
Image.fromarray(tint).resize(((X1 - X0) * Z, (Y1 - Y0) * Z), Image.NEAREST).save(
    OUT / "strip_parsed_10x.png"
)
print("classes present:", {int(c): int((labels[Y0:Y1, X0:X1] == c).sum()) for c in np.unique(labels[Y0:Y1, X0:X1])})
print("wrote", (X1 - X0) * Z, (Y1 - Y0) * Z)
