"""Where the old white knit collar survives in each render (neck band)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-submit-run"
ORIG = ROOT / "data/inputs/IMG_3651.jpg"
PKG = ROOT / "artifacts/local-edit-package/20261009T-neckfix"
RENDERS = {
    "final": OUT / "final.png",
    "clothing_pass": OUT / "clothing_pass.png",
    "prev_ghost_failure": ROOT / "artifacts/p29-fix-run/result_0.png",
    "prev_second_face_failure": ROOT / "artifacts/p29-run/result_0.png",
}
Y0, Y1, X0, X1 = 520, 1000, 240, 820
PALE = 165
KNIT = 150


def main() -> int:
    orig_img = Image.open(ORIG).convert("RGB")
    orig_l = np.asarray(orig_img.convert("L")).astype(np.int16)
    orig_sat = np.asarray(orig_img.convert("HSV"))[..., 1].astype(np.int16)
    cloth = np.asarray(Image.open(PKG / "clothing_edit_mask.png").convert("L")) > 127
    prot = np.asarray(Image.open(PKG / "protected_identity_mask.png").convert("L")) > 127
    band = np.zeros(orig_l.shape, bool)
    band[Y0:Y1, X0:X1] = True
    knit = (orig_l > KNIT) & band
    report = {
        "band": {"y": [Y0, Y1], "x": [X0, X1]},
        "original_pale_knit_pixels": int(knit.sum()),
        "original_knit_mean_rgb": [
            round(float(v), 1) for v in np.asarray(orig_img)[knit].reshape(-1, 3).mean(axis=0)
        ],
        "original_knit_desaturated_share": round(
            float((orig_sat[knit] < 40).mean()), 4
        ),
        "renders": {},
    }
    for name, path in RENDERS.items():
        res_img = Image.open(path).convert("RGB")
        res = np.asarray(res_img)
        res_l = np.asarray(res_img.convert("L")).astype(np.int16)
        pale = (res_l > PALE) & knit
        entry = {"pale_where_original_knit": int(pale.sum()),
                 "share_of_original_knit": round(float(pale.sum() / max(knit.sum(), 1)), 4)}
        if pale.sum():
            entry["mean_rgb_where_survived"] = [
                round(float(v), 1) for v in res[pale].reshape(-1, 3).mean(axis=0)
            ]
            entry["still_desaturated_share"] = round(
                float((np.asarray(Image.fromarray(res).convert("HSV"))[..., 1][pale] < 40).mean()), 4
            )
        for region_name, region in (("inside_clothing_mask", cloth), ("inside_protected_mask", prot)):
            sel = pale & region
            entry[f"{region_name}_pixels"] = int(sel.sum())
        count, labels_out = cv2.connectedComponents((pale & cloth).astype(np.uint8), connectivity=8)
        largest, area, box = 0, 0, None
        for i in range(1, count):
            size = int((labels_out == i).sum())
            if size > area:
                area, largest = size, i
        if largest:
            ys, xs = np.where(labels_out == largest)
            box = {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max())}
        entry["largest_ghost_blob_in_clothing"] = {"pixels": area, "bbox": box}
        report["renders"][name] = entry
    (OUT / "ghost_measure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
