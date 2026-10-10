"""Preview: current collar_repair_mask over the best final, 2x zoom on the collar."""
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB"))
mask = np.asarray(Image.open(REPO / "evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))

over = final.copy()
over[mask > 0] = (0.45 * over[mask > 0] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)

def crop2x(img):
    c = img[560:760, 400:640]
    im = Image.fromarray(c).resize((c.shape[1] * 2, c.shape[0] * 2), Image.NEAREST)
    return im

side = np.concatenate([crop2x(final), crop2x(over)], axis=1)
Image.fromarray(side).save(REPO / "evidence/p29-collar-repair/mask_preview_zoom2x.png")
print("saved", (side.shape[1], side.shape[0]))
print("mask px", int((mask > 0).sum()))
