"""Byte-exact identity check for the core-identity pixels of a real render.

The earlier runs only reported "within tolerance" numbers (identical_le2 /
identical_le11), which is not the same claim as "the original pixels survived".
This script crosses the original photo with the render and reports, for every
region that must not change, the share of pixels that are *byte identical*
versus merely close, plus the worst per-channel delta.

Offline only: reads the original, the shipped/fixed masks and one render.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.domains.portrait import human_parsing as hp  # noqa: E402
from app.domains.portrait.edit_mask import build_semantic_edit_masks  # noqa: E402
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
SHIPPED = ROOT / "artifacts/local-edit-package/20261008T153609Z"
OUT = ROOT / "evidence/p29-neck-fix"
GATE = ROOT / "artifacts/p29-fix-run/gate_report.json"


def stitch_crop_regions(image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Pixels outside the CropAndStitch crop boxes of the last real render.

    ``InpaintCropImproved`` downscales its crop to 1024; ``InpaintStitchImproved``
    scales it back up, so every pixel *inside* the box is resampled once. Pixels
    outside the boxes never leave the image and are the control for "did the
    pipeline touch this at all".
    """
    if not GATE.is_file():
        return {}
    geometry = json.loads(GATE.read_text())["mask_geometry"]
    height, width = image_shape
    rows, cols = np.mgrid[0:height, 0:width]

    def inside(entry: dict) -> np.ndarray:
        crop = entry["crop"]
        x0, y0 = crop["x"], crop["y"]
        x1, y1 = x0 + crop["w"], y0 + crop["h"]
        return (cols >= x0) & (cols < x1) & (rows >= y0) & (rows < y1)

    hair_box = inside(geometry["hair"])
    clothing_box = inside(geometry["clothing"])
    return {
        "outside_both_stitch_crops": ~(hair_box | clothing_box),
        "outside_hair_stitch_crop": ~hair_box,
    }


def region_stats(orig: np.ndarray, res: np.ndarray, region: np.ndarray) -> dict:
    n = int(region.sum())
    if n == 0:
        return {"pixels": 0}
    a = orig[region].astype(np.int16)
    b = res[region].astype(np.int16)
    d = np.abs(a - b)
    dmax = d.max(axis=1)
    return {
        "pixels": n,
        "byte_identical": round(float((dmax == 0).mean()), 6),
        "identical_le2": round(float((dmax <= 2).mean()), 6),
        "identical_le11": round(float((dmax <= 11).mean()), 6),
        "changed_gt25": int((dmax > 25).sum()),
        "mean_abs_delta": round(float(d.max(axis=1).mean()), 3),
        "max_abs_delta": int(d.max()),
        "p99_abs_delta": int(np.percentile(dmax, 99)),
    }


def main() -> int:
    image = np.asarray(Image.open(ORIG).convert("RGB"))
    labels = np.asarray(Image.open(LABELS))
    svc = NativeMaskInpainting(settings)
    landmarks = svc._detect_landmarks(ORIG, image)

    masks = build_semantic_edit_masks(image, landmarks, labels)
    shipped_prot = np.asarray(Image.open(SHIPPED / "protected_identity_mask.png").convert("L")) > 127
    fixed_prot = np.asarray(masks.protected) > 0
    fixed_core = np.asarray(masks.face_core) > 0
    fixed_hair = np.asarray(masks.hair) > 0
    parse_face = labels == hp.FACE
    parse_hair = labels == hp.HAIR
    background = labels == hp.BACKGROUND

    regions = {
        "protected_shipped": shipped_prot,
        "protected_fixed": fixed_prot,
        "facial_core_fixed": fixed_core,
        "protected_minus_facial_core": fixed_prot & ~fixed_core,
        "parsed_face_all": parse_face,
        "parsed_hair_inside_fixed_hair": parse_hair & fixed_hair,
        "parsed_hair_inside_fixed_protected": parse_hair & fixed_prot,
        "background_all": background,
        "background_inside_editable": background
        & (fixed_hair | (np.asarray(masks.clothing) > 0)),
        "outside_all_masks": ~(
            shipped_prot | fixed_prot | fixed_core | fixed_hair
            | (np.asarray(masks.clothing) > 0)
        ),
    }
    regions.update(stitch_crop_regions(image.shape[:2]))

    renders = {
        "p29_fix_run_2108231348232871937": ROOT / "artifacts/p29-fix-run/result_0.png",
        "p29_run_2108222523123662850": ROOT / "artifacts/p29-run/result_0.png",
    }

    report: dict = {
        "question": (
            "are the core identity pixels byte-identical to the original, or "
            "only within a tolerance?"
        ),
        "metric": (
            "byte_identical = max per-channel |delta| == 0; identical_le2/le11 = "
            "the tolerance bands the earlier reports used"
        ),
    }
    for name, path in renders.items():
        if not path.is_file():
            report[name] = "missing"
            continue
        res = np.asarray(Image.open(path).convert("RGB"))
        if res.shape != image.shape:
            report[name] = f"size mismatch {res.shape} vs {image.shape}"
            continue
        report[name] = {
            region: region_stats(image, res, mask) for region, mask in regions.items()
        }

    (OUT / "pixel_identity_strict.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
