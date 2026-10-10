"""Before/after mask visuals for the neck / hairline / collar fix.

Offline only: reuses the cached SCHP label map and the shipped package masks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.edit_mask import (  # noqa: E402
    SemanticMaskConfig,
    build_semantic_edit_masks,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

PKG = ROOT / "artifacts/local-edit-package/20261008T153609Z"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
FAIL = ROOT / "artifacts/p29-fix-run/result_0.png"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
OUT = ROOT / "evidence/p29-neck-fix"

image = np.asarray(Image.open(ORIG).convert("RGB"))
fail = np.asarray(Image.open(FAIL).convert("RGB"))
labels = np.asarray(Image.open(LABELS))
svc = NativeMaskInpainting(settings)
lm = svc._detect_landmarks(ORIG, image)

after = build_semantic_edit_masks(image, lm, labels)
variants = {
    "default": after,
    # The release was briefly tuned to a one-pixel erode before being settled at
    # zero; kept as the comparison point so the panel shows what the knob does.
    "erode001": build_semantic_edit_masks(
        image, lm, labels, config=SemanticMaskConfig(hair_guard_release_erode_ratio=0.01)
    ),
}
before = {
    "hair": np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127,
    "cloth": np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127,
    "prot": np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127,
}

parsed_hair = labels == hp.HAIR
frozen = {}
for name, ms in variants.items():
    prot = np.asarray(ms.protected) > 0
    hair = np.asarray(ms.hair) > 0
    frozen[name] = {
        "hairFrozenByGuard": int((prot & parsed_hair).sum()),
        "hairPixels": int(hair.sum()),
        "hairAndCloth": int((hair & (np.asarray(ms.clothing) > 0)).sum()),
        "hairOverClothing": int((hair & np.isin(labels, hp.CLOTHING_CLASSES)).sum()),
        "editablePixels": int(ms.stats["editablePixels"]),
        "backgroundShare": ms.stats["backgroundEditableShare"],
        "exact_pixel_overlap_with_core": int(
            (hair & (np.asarray(ms.face_core) > 0)).sum()
        ),
    }
(OUT / "guard_release_variants.json").write_text(json.dumps(frozen, indent=1))


def outline(mask: np.ndarray, rgb, width=2):
    m = (mask * 255).astype(np.uint8)
    edge = (m != np.asarray(__import__("cv2").erode(m, np.ones((3, 3), np.uint8)))) & mask
    out = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    out[edge] = rgb
    return out


import cv2  # noqa: E402

HAIR_C = (255, 64, 64)
CLOTH_C = (70, 140, 255)
PROT_C = (255, 255, 255)


def overlay(photo, masks, *, alpha=0.45):
    out = photo.astype(np.float32).copy()
    for mask, colour in masks:
        m = mask[..., None].astype(np.float32)
        out = out * (1 - m * alpha) + np.array(colour, np.float32) * (m * alpha)
    return out.astype(np.uint8)


def label_strip(pil_img, text):
    im = pil_img.convert("RGB")
    strip = Image.new("RGB", (im.width, 26), (16, 16, 16))
    ImageDraw.Draw(strip).text((6, 6), text, fill=(235, 235, 235))
    canvas = Image.new("RGB", (im.width, im.height + 26), (16, 16, 16))
    canvas.paste(strip, (0, 0))
    canvas.paste(im, (0, 26))
    return canvas


def hstack(images, gap=8, bg=(16, 16, 16)):
    h = max(i.height for i in images)
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for i in images:
        canvas.paste(i, (x, 0))
        x += i.width + gap
    return canvas


def crop_box(img, box):
    return img.crop((box[1], box[0], box[3], box[2]))  # box = y0,y1,x0,x1


def mask_image(mask, name):
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / name)


# 1. hair mask, before vs after, over the original
hair_before_img = overlay(image, [(before["hair"], HAIR_C)])
hair_after_img = overlay(image, [(np.asarray(variants["default"].hair) > 0, HAIR_C)])
hair_erode_img = overlay(image, [(np.asarray(variants["erode001"].hair) > 0, HAIR_C)])
panel = hstack([
    label_strip(Image.fromarray(image), "ORIGINAL"),
    label_strip(Image.fromarray(hair_before_img), f"HAIR - SHIPPED ({int(before['hair'].sum()):,} px)"),
    label_strip(Image.fromarray(hair_after_img), f"HAIR - FIXED ({frozen['default']['hairPixels']:,} px)"),
    label_strip(Image.fromarray(hair_erode_img), f"HAIR - FIXED erode .01 ({frozen['erode001']['hairPixels']:,} px)"),
])
panel.save(OUT / "hair_mask_before_after.png")

# 2. clothing mask, before vs after
cloth_before_img = overlay(image, [(before["cloth"], CLOTH_C)])
cloth_after_img = overlay(image, [(np.asarray(after.clothing) > 0, CLOTH_C)])
panel = hstack([
    label_strip(Image.fromarray(image), "ORIGINAL"),
    label_strip(Image.fromarray(cloth_before_img), f"CLOTHING - SHIPPED ({int(before['cloth'].sum()):,} px)"),
    label_strip(Image.fromarray(cloth_after_img), f"CLOTHING - FIXED ({after.stats['clothingPixels']:,} px)"),
    label_strip(
        Image.fromarray(overlay(image, [
            (before["cloth"] & parsed_hair, (255, 220, 60)),
        ])),
        f"SHIPPED CLOTH OVER PARSED HAIR ({(before['cloth'] & parsed_hair).sum():,} px)",
    ),
])
panel.save(OUT / "cloth_mask_before_after.png")

# 3. neck ghost map on the real failure render
diff = np.abs(image.astype(np.int16) - fail.astype(np.int16)).max(axis=2)
res_l = np.asarray(Image.open(FAIL).convert("L")).astype(np.int16)
orig_l = np.asarray(Image.open(ORIG).convert("L")).astype(np.int16)
ghost = (diff > 25) & ((res_l - orig_l) > 15)
neck = np.zeros_like(ghost)
neck[540:1010, 210:850] = True
ghost_img = overlay(fail, [(ghost & neck, (255, 0, 255)), (before["hair"] & neck, HAIR_C)])
ghost_img = overlay(ghost_img, [(ghost & neck, (255, 0, 255))])
panel = hstack([
    label_strip(Image.fromarray(image), "ORIGINAL"),
    label_strip(Image.fromarray(fail), "FAILED RENDER (task ...850)"),
    label_strip(
        Image.fromarray(overlay(fail, [(ghost, (255, 0, 255))])),
        "PALE GHOST (37,922 px) in magenta",
    ),
    label_strip(
        Image.fromarray(overlay(fail, [
            (ghost & before["hair"], (255, 0, 255)),
            (ghost & before["cloth"], (0, 200, 120)),
            (ghost & before["prot"], (255, 255, 0)),
            (ghost & ~(before["hair"] | before["cloth"] | before["prot"]), (255, 255, 255)),
        ])),
        "GHOST vs SHIPPED MASKS: hair=magenta cloth=green prot=yellow",
    ),
])
panel.save(OUT / "neck_ghost_map.png")

# 4. hairline band, 3x, before/after masks + protected, with the frozen fringe
box_y = (60, 420)  # rows of the hairline band
box_x = (200, 720)  # columns of the hairline band


def zoom(img_pil, factor=3):
    return img_pil.resize(
        (img_pil.width * factor, img_pil.height * factor), Image.NEAREST
    )


def band(arr):
    return arr[box_y[0]:box_y[1], box_x[0]:box_x[1]]


prot_after = np.asarray(after.protected) > 0
hair_after = np.asarray(after.hair) > 0
band_layers = overlay(band(image), [
    (band(before["hair"]), HAIR_C),
    (band(before["prot"]), PROT_C),
])
band_layers_after = overlay(band(image), [
    (band(hair_after), HAIR_C),
    (band(prot_after), PROT_C),
])
frozen_before = band(before["prot"] & parsed_hair)
frozen_img = band(image).copy()
frozen_img[frozen_before] = (255, 0, 255)
frozen_after = band(prot_after & parsed_hair)
frozen_after_img = band(image).copy()
frozen_after_img[frozen_after] = (255, 0, 255)
panel = hstack([
    label_strip(zoom(Image.fromarray(band(image))), "HAIRLINE BAND 3x - ORIGINAL"),
    label_strip(zoom(Image.fromarray(band_layers)), "SHIPPED: hair red / protect white"),
    label_strip(
        zoom(Image.fromarray(frozen_img)),
        f"SHIPPED frozen fringe ({int(frozen_before.sum()):,} px) magenta",
    ),
    label_strip(zoom(Image.fromarray(band_layers_after)), "FIXED: hair red / protect white"),
    label_strip(
        zoom(Image.fromarray(frozen_after_img)),
        f"FIXED frozen fringe ({int(frozen_after.sum()):,} px) magenta",
    ),
])
panel.save(OUT / "hairline_mask_zoom3x.png")

# 5. three-layer overlay, before vs after
def three_layer(ms):
    return overlay(image, [
        (np.asarray(ms.hair) > 0, HAIR_C),
        (np.asarray(ms.clothing) > 0, CLOTH_C),
        (np.asarray(ms.protected) > 0, PROT_C),
    ])


class Shipped:
    hair, clothing, protected = before["hair"], before["cloth"], before["prot"]


panel = hstack([
    label_strip(Image.fromarray(image), "ORIGINAL"),
    label_strip(Image.fromarray(three_layer(Shipped)), "SHIPPED hair/red cloth/blue protect/white"),
    label_strip(Image.fromarray(three_layer(after)), "FIXED hair/red cloth/blue protect/white"),
])
panel.save(OUT / "layers_before_after.png")

mask_image(before["hair"], "mask_before_hair.png")
mask_image(np.asarray(after.hair) > 0, "mask_after_hair.png")
mask_image(before["cloth"], "mask_before_cloth.png")
mask_image(np.asarray(after.clothing) > 0, "mask_after_cloth.png")
mask_image(before["prot"], "mask_before_prot.png")
mask_image(prot_after, "mask_after_prot.png")
print(json.dumps(frozen, indent=1))
