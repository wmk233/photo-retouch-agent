"""Acceptance checks for the single authorized paid render (task 2108330225116082177).

Offline only: reads the original photo, the shipped masks, the cached SCHP label
map and the three renders (two earlier failures + this run). No network, no key,
no further task.

Regions come from the shipped package the task actually used, and the identity
guards are rebuilt from the same MediaPipe landmarks the pipeline used, so the
check is independent of the render but not of the masks.
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
from app.domains.portrait.edit_mask import (  # noqa: E402
    GLASSES_BAND,
    LIPS_INNER,
    SemanticMaskConfig,
    _hull,
    build_semantic_edit_masks,
    dilate_mask,
    face_width,
    fill_feature_guard,
    fill_polygon,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

OUT = ROOT / "evidence/p29-submit-run"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
LABELS = ROOT / "evidence/p29-neck-fix/p29_labels.png"
PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
GATE = ROOT / "artifacts/p29-submit-run/gate_report.json"

RENDERS = {
    "final": OUT / "final.png",
    "clothing_pass": OUT / "clothing_pass.png",
    "prev_ghost_failure": ROOT / "artifacts/p29-fix-run/result_0.png",
    "prev_second_face_failure": ROOT / "artifacts/p29-run/result_0.png",
}

# Bands used by the previous offline forensics, unchanged so the numbers compare.
NECK_BAND = (slice(520, 1000), slice(240, 820))
HAIRLINE_BAND = (slice(60, 420), slice(200, 720))
CHEST_BAND = (slice(700, 1050), slice(300, 610))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def region_stats(orig: np.ndarray, res: np.ndarray, region: np.ndarray) -> dict:
    n = int(region.sum())
    if n == 0:
        return {"pixels": 0}
    d = np.abs(orig[region].astype(np.int16) - res[region].astype(np.int16))
    dmax = d.max(axis=1)
    return {
        "pixels": n,
        "byte_identical": round(float((dmax == 0).mean()), 6),
        "identical_le2": round(float((dmax <= 2).mean()), 6),
        "changed_gt25": int((dmax > 25).sum()),
        "max_abs_delta": int(d.max()),
        "mean_abs_delta": round(float(dmax.mean()), 3),
    }


def best_shift(orig: np.ndarray, res: np.ndarray, region: np.ndarray, limit: int = 4) -> dict:
    """Integer translation that best explains the render outside the edited area."""
    base = orig[region].astype(np.int16)
    target = res[region].astype(np.int16)
    height, width = orig.shape[:2]
    rows, cols = np.mgrid[0:height, 0:width]
    best = None
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            sr = np.clip(rows + dy, 0, height - 1)
            sc = np.clip(cols + dx, 0, width - 1)
            shifted = orig[sr, sc][region].astype(np.int16)
            score = float(np.abs(shifted - target).mean())
            if best is None or score < best[0]:
                best = (score, dx, dy)
    zero = float(np.abs(base - target).mean())
    return {"best_dx": best[1], "best_dy": best[2], "best_mae": round(best[0], 3),
            "zero_shift_mae": round(zero, 3)}


def face_count(path: Path) -> dict:
    """How many faces MediaPipe finds. A second painted head shows up here."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(ROOT / "data/models/face_landmarker.task")
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.3,
    )
    with vision.FaceLandmarker.create_from_options(options) as lm:
        image = mp.Image.create_from_file(str(path))
        result = lm.detect(image)
    boxes = []
    for face in result.face_landmarks:
        xs = [p.x for p in face]
        ys = [p.y for p in face]
        boxes.append([round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)])
    return {"faces": len(boxes), "boxes_normalized": boxes}


def main() -> int:
    orig_img = Image.open(ORIG).convert("RGB")
    orig = np.asarray(orig_img)
    labels = np.asarray(Image.open(LABELS))
    svc = NativeMaskInpainting(settings)
    lm = svc._detect_landmarks(ORIG, orig)
    masks = build_semantic_edit_masks(orig, lm, labels)
    shape = orig.shape[:2]
    face_w = face_width(lm)

    features = fill_feature_guard(shape, lm)
    features = np.maximum(features, fill_polygon(shape, _hull(lm[list(LIPS_INNER)])))
    features = dilate_mask(features, SemanticMaskConfig().feature_grow_ratio * face_w)
    glasses = dilate_mask(fill_polygon(shape, _hull(lm[list(GLASSES_BAND)])),
                          SemanticMaskConfig().glasses_grow_ratio * face_w)

    regions = {
        "protected_identity": np.asarray(masks.protected) > 0,
        "facial_core": np.asarray(masks.face_core) > 0,
        "feature_guard_eyes_nose_mouth_teeth": np.asarray(features) > 0,
        "glasses_guard": np.asarray(glasses) > 0,
        "hair_edit_mask": np.asarray(masks.hair) > 0,
        "clothing_edit_mask": np.asarray(masks.clothing) > 0,
        "parsed_background": labels == hp.BACKGROUND,
    }
    ship = {
        "hair_edit_mask": np.asarray(Image.open(PKG / "hair_edit_mask.png").convert("L")) > 127,
        "clothing_edit_mask": np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127,
        "protected_identity": np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127,
    }
    mask_match = {
        k: round(float((regions[k] == v).mean()), 6) for k, v in ship.items()
    }

    geometry = json.loads(GATE.read_text())["mask_geometry"]
    height, width = shape
    rows, cols = np.mgrid[0:height, 0:width]

    def box(entry: dict) -> np.ndarray:
        crop = entry["crop"]
        return ((cols >= crop["x"]) & (cols < crop["x"] + crop["w"])
                & (rows >= crop["y"]) & (rows < crop["y"] + crop["h"]))

    outside_crops = ~(box(geometry["hair"]) | box(geometry["clothing"]))
    regions["outside_both_stitch_crops"] = outside_crops

    orig_l = np.asarray(orig_img.convert("L")).astype(np.int16)
    report: dict = {
        "task_id": "2108330225116082177",
        "regions": {k: int(v.sum()) for k, v in regions.items()},
        "mask_match_vs_shipped_package": mask_match,
        "renders": {},
    }

    for name, path in RENDERS.items():
        if not path.is_file():
            report["renders"][name] = {"missing": str(path)}
            continue
        res_img = Image.open(path).convert("RGB")
        res = np.asarray(res_img)
        res_l = np.asarray(res_img.convert("L")).astype(np.int16)
        entry: dict = {
            "path": str(path.relative_to(ROOT)),
            "size": list(res_img.size),
            "size_matches_original": list(res_img.size) == list(orig_img.size),
            "shift": best_shift(orig, res, outside_crops),
        }
        entry["regions"] = {
            key: region_stats(orig, res, mask) for key, mask in regions.items()
        }
        delta = np.abs(orig.astype(np.int16) - res.astype(np.int16)).max(axis=2)
        brighter = res_l.astype(np.int16) - orig_l
        neck = np.zeros(shape, bool)
        neck[NECK_BAND] = True
        hairline = np.zeros(shape, bool)
        hairline[HAIRLINE_BAND] = True
        chest = np.zeros(shape, bool)
        chest[CHEST_BAND] = True

        ghost = (delta > 25) & (brighter > 15) & neck
        entry["neck_band"] = {
            "pixels": int(neck.sum()),
            "ghost_pixels": int(ghost.sum()),
            "ghost_in_protected": int((ghost & regions["protected_identity"]).sum()),
            "ghost_in_clothing_mask": int((ghost & regions["clothing_edit_mask"]).sum()),
            "ghost_in_hair_mask": int((ghost & regions["hair_edit_mask"]).sum()),
            "ghost_on_parsed_clothing": int((ghost & (np.isin(labels, hp.CLOTHING_CLASSES))).sum()),
            "ghost_on_parsed_background": int((ghost & regions["parsed_background"]).sum()),
        }
        # The old white knit collar: pale pixels the original also had pale.
        pale = (res_l > 165) & (orig_l > 150) & neck
        entry["neck_band"]["pale_remnant_pixels"] = int(pale.sum())
        entry["neck_band"]["pale_remnant_in_protected"] = int((pale & regions["protected_identity"]).sum())
        entry["neck_band"]["pale_remnant_in_clothing_mask"] = int((pale & regions["clothing_edit_mask"]).sum())
        # A deep V would put skin where the original had knit.
        skin_ref = orig[regions["facial_core"]].astype(np.float32)
        skin_mean = skin_ref.reshape(-1, 3).mean(axis=0)
        chest_skin = chest & (np.linalg.norm(res.astype(np.float32) - skin_mean, axis=2) < 55)
        entry["chest_band"] = {
            "pixels": int(chest.sum()),
            "skin_like_pixels_in_final": int(chest_skin.sum()),
            "skin_like_pixels_in_original": int(
                (chest & (np.linalg.norm(orig.astype(np.float32) - skin_mean, axis=2) < 55)).sum()
            ),
            "skin_reference_rgb": [round(float(v), 1) for v in skin_mean],
        }
        # Hairline: strong horizontal luminance edges inside the band.
        grad = np.abs(np.diff(res_l, axis=0))
        grad_o = np.abs(np.diff(orig_l, axis=0))
        entry["hairline_band"] = {
            "pixels": int(hairline.sum()),
            "changed_gt25": int((delta[hairline] > 25).sum()),
            "strong_edge_rows_final": int((grad[HAIRLINE_BAND[0].start:HAIRLINE_BAND[0].stop, HAIRLINE_BAND[1]] > 40).sum()),
            "strong_edge_rows_original": int((grad_o[HAIRLINE_BAND[0].start:HAIRLINE_BAND[0].stop, HAIRLINE_BAND[1]] > 40).sum()),
        }
        entry["faces"] = face_count(path)
        report["renders"][name] = entry

    (OUT / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
