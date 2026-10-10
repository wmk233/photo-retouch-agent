from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
orig = np.asarray(Image.open(REPO/"data/inputs/IMG_3651.jpg").convert("RGB"))
c = orig[570:740, 450:650]
Image.fromarray(c).resize((c.shape[1]*5, c.shape[0]*5), Image.LANCZOS).save(REPO/"evidence/p29-collar-repair/orig_collar_zoom5x.png")
print(c.shape)
