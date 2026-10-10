"""Round-3 collar fix verification: old knit-collar ghost + AI pale lining source."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-collar-final-run"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"

BEFORE_PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
AFTER_PKG = ROOT / "artifacts/local-edit-package/20261009T-collarfix"

RENDERS = {
    "prev_clothing_pass": ROOT / "evidence/p29-submit-run/clothing_pass.png",
    "prev_final": ROOT / "evidence/p29-submit-run/final.png",
    "new_clothing_pass": ROOT / "evidence/p29-collar-final-run/clothing_pass.png",
    "new_final": ROOT / "evidence/p29-collar-final-run/final.png",
}

Y0, Y1, X0, X1 = 520, 1000, 240, 820
PALE = 165
KNIT = 150
STRIP = (594, 729, 478, 562)
FACE = (120, 620, 250, 660)
SEP = 8


def _load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def _gray(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L")).astype(np.int16)


def _label_bar(label: str, width: int, height: int = 30) -> np.ndarray:
    bar = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(bar, label, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def tile(items, scale: int = 1) -> Image.Image:
    tiles = []
    for label, img in items:
        arr = np.asarray(img) if isinstance(img, np.ndarray) else np.asarray(img.convert("RGB"))
        arr = np.ascontiguousarray(arr)
        if scale != 1:
            arr = cv2.resize(arr, (arr.shape[1] * scale, arr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        tiles.append(np.vstack([_label_bar(label, arr.shape[1]), arr]))
    width = sum(t.shape[1] for t in tiles) + SEP * (len(tiles) - 1)
    height = max(t.shape[0] for t in tiles)
    canvas = np.full((height, width, 3), 255, np.uint8)
    x = 0
    for t in tiles:
        canvas[0:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + SEP
    return Image.fromarray(canvas)


def crop(arr: np.ndarray, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    return arr[y0:y1, x0:x1]


def main() -> int:
    orig_arr = _load(ORIG)
    orig_l = _gray(ORIG)

    before_prot = np.asarray(Image.open(BEFORE_PKG / "protected_identity_mask.png").convert("L")) > 127
    after_prot = np.asarray(Image.open(AFTER_PKG / "protected_identity_mask.png").convert("L")) > 127
    clothing = np.asarray(Image.open(AFTER_PKG / "clothing_edit_mask.png").convert("L")) > 127
    hair = np.asarray(Image.open(AFTER_PKG / "hair_edit_mask.png").convert("L")) > 127

    band = np.zeros(orig_l.shape, bool)
    band[Y0:Y1, X0:X1] = True
    knit = (orig_l > KNIT) & band

    sy0, sy1, sx0, sx1 = STRIP
    strip = np.zeros(orig_l.shape, bool)
    strip[sy0:sy1, sx0:sx1] = True
    face_rect = np.zeros(orig_l.shape, bool)
    fy0, fy1, fx0, fx1 = FACE
    face_rect[fy0:fy1, fx0:fx1] = True

    released = before_prot & ~after_prot
    kept = strip & ~released

    report = {
        "band": {"y": [Y0, Y1], "x": [X0, X1]},
        "original_pale_knit_pixels": int(knit.sum()),
        "mask_delta": {
            "before_protected": int(before_prot.sum()),
            "after_protected": int(after_prot.sum()),
            "released_total": int(released.sum()),
            "clothing_total": int(clothing.sum()),
            "hair_total": int(hair.sum()),
            "hair_and_clothing": int((hair & clothing).sum()),
            "freed_over_skin": int((released & (orig_l > 90) & (orig_l < 210)).sum()),
        },
        "strip": {
            "box": [sy0, sy1, sx0, sx1],
            "strip_pixels": int(strip.sum()),
            "strip_released": int((strip & released).sum()),
            "strip_still_protected": int((strip & after_prot).sum()),
            "kept_not_released": int(kept.sum()),
            "kept_pale_in_original": int((kept & (orig_l > KNIT)).sum()),
        },
        "renders": {},
        "prev_vs_new": {},
    }

    for name, path in RENDERS.items():
        res_arr = _load(path)
        res_l = _gray(path)
        pale = (res_l > PALE) & knit
        entry = {
            "pale_where_original_knit": int(pale.sum()),
            "share_of_original_knit": round(float(pale.sum() / max(knit.sum(), 1)), 4),
        }
        if pale.sum():
            entry["mean_rgb_where_survived"] = [round(float(v), 1) for v in res_arr[pale].reshape(-1, 3).mean(axis=0)]
        for region_name, region in (("inside_clothing_mask", clothing), ("inside_protected_mask", after_prot)):
            entry[region_name + "_pixels"] = int((pale & region).sum())
        count, labels_out = cv2.connectedComponents((pale & clothing).astype(np.uint8), connectivity=8)
        largest, area, box = 0, 0, None
        for i in range(1, count):
            size = int((labels_out == i).sum())
            if size > area:
                area, largest = size, i
        if largest:
            ys, xs = np.where(labels_out == largest)
            box = {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max())}
        entry["largest_ghost_blob_in_clothing"] = {"pixels": area, "bbox": box}

        eq = np.all(res_arr == orig_arr, axis=2)
        entry["fidelity"] = {
            "old_guard_share_identical_to_original": round(float(eq[before_prot].mean()), 4) if before_prot.sum() else None,
            "face_rect_share_identical": round(float(eq[face_rect].mean()), 4),
            "outside_edit_share_identical": round(float(eq[~(clothing | hair)].mean()), 4),
        }
        diff_abs = np.abs(res_arr.astype(np.int16) - orig_arr.astype(np.int16))
        entry["strip_metrics"] = {
            "mean_luma": round(float(res_l[strip].mean()), 1),
            "pale_pixels": int((res_l[strip] > PALE).sum()),
            "mean_abs_diff_vs_original": round(float(diff_abs[strip].mean()), 2),
        }
        report["renders"][name] = entry

    prev_arr = _load(RENDERS["prev_final"])
    new_arr = _load(RENDERS["new_final"])
    d = np.abs(prev_arr.astype(np.int16) - new_arr.astype(np.int16)).sum(axis=2)
    report["prev_vs_new"] = {
        "mean_abs_diff_sum_rgb": round(float(d.mean()), 2),
        "changed_pixels_gt30": int((d > 30).sum()),
        "changed_inside_protected_mask": int(((d > 30) & after_prot).sum()),
        "changed_outside_edit": int(((d > 30) & ~(clothing | hair)).sum()),
        "prev_final_abs_diff_vs_original": round(float(np.abs(prev_arr.astype(np.int16) - orig_arr.astype(np.int16)).mean()), 2),
        "new_final_abs_diff_vs_original": round(float(np.abs(new_arr.astype(np.int16) - orig_arr.astype(np.int16)).mean()), 2),
    }

    t = tile([("ORIGINAL", orig_arr), ("PREV FINAL", prev_arr), ("NEW CLOTHING PASS", _load(RENDERS["new_clothing_pass"])), ("NEW FINAL", new_arr)])
    t.save(OUT / "four_up_1x.png")

    neck_box = (545, 800, 420, 660)
    tile([
        ("ORIGINAL", crop(orig_arr, neck_box)),
        ("PREV FINAL", crop(prev_arr, neck_box)),
        ("NEW FINAL", crop(new_arr, neck_box)),
    ], scale=8).save(OUT / "neck_8x_orig_prev_new.png")

    tile([
        ("NEW CLOTHING PASS", crop(_load(RENDERS["new_clothing_pass"]), neck_box)),
        ("NEW FINAL", crop(new_arr, neck_box)),
    ], scale=8).save(OUT / "neck_8x_clothingpass_vs_final.png")

    tile([
        ("ORIGINAL", crop(orig_arr, FACE)),
        ("PREV FINAL", crop(prev_arr, FACE)),
        ("NEW FINAL", crop(new_arr, FACE)),
    ], scale=3).save(OUT / "face_hairline_3x.png")

    dnorm = np.clip(d, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat).save(OUT / "diff_prev_vs_new.png")

    (OUT / "compare_renders.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
