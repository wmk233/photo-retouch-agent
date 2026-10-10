"""Render the local edit masks for one photo as reviewable PNGs.

Usage (from ``backend/``)::

    ../.venv/bin/python -m tools.build_edit_masks
    ../.venv/bin/python -m tools.build_edit_masks --image data/inputs/IMG_3651.jpg --out-dir ../evidence

This tool only *measures* geometry. It does not generate pixels, call any paid
provider, download any weight or train anything -- it reads the photo, runs the
already-installed MediaPipe face landmark model and draws the mask the inpainting
pipeline would later feed to a generator.

What the pictures mean
----------------------
* ``p26_original.png``  -- the untouched photograph.
* ``p26_protected.png`` -- red = pixels a generator must return unchanged
  (eyes, brows, nose, lips, teeth, the subject's own skin). The rest is dimmed.
* ``p26_editable.png``  -- green = pixels a generator may repaint (the hair band
  and the clothing band). The rest is dimmed.
* ``p26_overlay.png``   -- both masks on the original, plus the face box.

Honesty about the hair/clothing bands: this environment has no hair matte and no
human-parsing model, so those two regions are an ellipse around the head and a
rectangle under the chin -- geometric heuristics, NOT segmentation results. The
gaps are written next to the images instead of being hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow ``python tools/build_edit_masks.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domains.portrait.edit_mask import build_edit_masks  # noqa: E402
from app.domains.portrait.face_lock import MediaPipeLandmarkDetector  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / "data" / "inputs" / "IMG_3651.jpg"),
        help="photo to measure",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "data" / "models" / "face_landmarker.task"),
        help="MediaPipe face landmarker task bundle",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "evidence"),
        help="directory the PNGs and stats JSON are written to",
    )
    parser.add_argument(
        "--landmarks-json",
        default=None,
        help="optional (N,2) landmark JSON to skip detection (offline review)",
    )
    parser.add_argument("--face-index", type=int, default=0, help="which detected face")
    return parser


def _load_landmarks(args: argparse.Namespace, image_rgb: np.ndarray) -> np.ndarray:
    if args.landmarks_json:
        with open(args.landmarks_json, "r", encoding="utf-8") as handle:
            return np.asarray(json.load(handle), dtype=np.float64)
    detector = MediaPipeLandmarkDetector(args.model)
    try:
        faces = detector.detect(image_rgb)
    finally:
        detector.close()
    if not faces:
        raise SystemExit("No face detected. Nothing to mask; no files were written.")
    if args.face_index >= len(faces):
        raise SystemExit(
            f"--face-index {args.face_index} out of range ({len(faces)} faces found)."
        )
    return np.asarray(faces[args.face_index], dtype=np.float64)


def _dim(image: np.ndarray, keep: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    """Darken everything outside *keep* so the live region reads clearly."""
    factor = np.where(keep > 0, 1.0, alpha)[..., None]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _tint(base: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], strength: float) -> np.ndarray:
    overlay = np.zeros_like(base, dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = colour
    blend = (mask > 0)[..., None].astype(np.float32) * strength
    out = base.astype(np.float32) * (1.0 - blend) + overlay * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def _save(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path} (no download is attempted).")
    if not args.landmarks_json and not Path(args.model).is_file():
        raise SystemExit(
            f"Face landmark model not found: {args.model}. "
            "Run tools.download_face_landmarker once, or pass --landmarks-json."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    landmarks = _load_landmarks(args, image_rgb)

    mask_set = build_edit_masks(image_rgb, landmarks)
    protected = mask_set.protected
    editable = mask_set.editable

    _save(out_dir / "p26_original.png", image_rgb)
    _save(out_dir / "p26_protected.png", _tint(_dim(image_rgb, protected), protected, (255, 40, 40), 0.55))
    _save(out_dir / "p26_editable.png", _tint(_dim(image_rgb, editable), editable, (40, 220, 80), 0.55))

    overlay = _dim(image_rgb, np.maximum(protected, editable), alpha=0.35)
    overlay = _tint(overlay, protected, (255, 40, 40), 0.45)
    overlay = _tint(overlay, editable, (40, 220, 80), 0.45)
    x0, y0, x1, y1 = mask_set.stats["faceBox"]
    cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (60, 160, 255), 2)
    _save(out_dir / "p26_overlay.png", overlay)

    stats_path = out_dir / "p26-edit-mask-stats.json"
    record = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "command": "python -m tools.build_edit_masks",
        "image": str(image_path),
        "model": str(args.model) if not args.landmarks_json else "landmarks-json",
        "paidGenerations": 0,
        "paidVisionCalls": 0,
        "note": (
            "Local geometry only. hair/clothing bands are heuristic ellipses and "
            "rectangles, NOT segmentation results. No generator was run."
        ),
        "stats": mask_set.stats,
        "gaps": list(mask_set.gaps),
        "outputs": [
            "p26_original.png",
            "p26_protected.png",
            "p26_editable.png",
            "p26_overlay.png",
        ],
    }
    stats_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote 4 PNGs + {stats_path.name} to {out_dir}")
    print(
        "  protected {protectedShare:.1%} | editable {editableShare:.1%} | "
        "identityCoversEditable={identityCoversEditable}".format(**mask_set.stats)
    )
    print(f"  hairPixels={mask_set.stats['hairPixels']} clothingPixels={mask_set.stats['clothingPixels']}")
    for gap in mask_set.gaps:
        print(f"  gap: {gap}")
    print("No generation, no paid provider, no download was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
