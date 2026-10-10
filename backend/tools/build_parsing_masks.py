"""Render the *semantic* P2.7 edit masks for one photo as reviewable PNGs.

Usage (from ``backend/``)::

    ../.venv/bin/python -m tools.build_parsing_masks --labels ../evidence/p27_labels.png

``--labels`` is the ATR class map that ``tools.run_human_parsing`` wrote with the
pretrained SCHP model; it is **required**. This tool never draws a stand-in
region: if the label map is missing or the wrong size it stops, and it never
falls back to the P2.6 ellipse while reporting success. That is the whole point
of P2.7 -- the hair region has to be a parsed silhouette, not a circle.

The mask semantics are the ones every mature inpainting stack uses
(``diffusers.AutoPipelineForInpainting``, BrushNet, the hosted endpoints):

    white (255) = repaint            black (0) = keep the original pixels

Everything is written in the photo's own coordinates and at the photo's own
size. Nothing here generates pixels, calls a paid provider or trains anything.
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

if __package__ in (None, ""):  # allow ``python tools/build_parsing_masks.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domains.portrait.edit_mask import (  # noqa: E402
    GLASSES_BAND,
    SemanticMaskConfig,
    build_edit_masks,
    build_semantic_edit_masks,
)
from app.domains.portrait.face_lock import (  # noqa: E402
    CHIN,
    FEATURE_GROUPS,
    LIPS_INNER,
    FaceLockConfig,
    MediaPipeLandmarkDetector,
    adaptive_skin_mask,
    dilate_mask,
    face_width,
    skin_reference_colour,
)
from app.domains.portrait.human_parsing import ATR_CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Review palette for the parsed classes, keyed by ATR id.
CLASS_COLOURS = {
    0: (0, 0, 0), 2: (255, 128, 0), 3: (255, 255, 0), 4: (0, 128, 255),
    7: (255, 0, 255), 11: (255, 192, 203), 14: (64, 255, 64), 15: (192, 0, 192),
}
DEFAULT_CLASS_COLOUR = (128, 128, 128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / "data" / "inputs" / "IMG_3651.jpg"),
        help="photo the label map belongs to",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="ATR class map PNG from tools.run_human_parsing (photo coordinates)",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "data" / "models" / "face_landmarker.task"),
        help="MediaPipe face landmarker task bundle",
    )
    parser.add_argument(
        "--landmarks-json",
        default=None,
        help="optional (N,2) landmark JSON to skip detection (offline review)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "evidence"),
        help="directory the PNGs and stats JSON are written to",
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
    factor = np.where(keep > 0, 1.0, alpha)[..., None]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _tint(base: np.ndarray, mask: np.ndarray, colour, strength: float) -> np.ndarray:
    overlay = np.zeros_like(base, dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = colour
    blend = (mask > 0)[..., None].astype(np.float32) * strength
    return np.clip(base.astype(np.float32) * (1.0 - blend) + overlay * blend, 0, 255).astype(np.uint8)


def _save(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _label(title: str, array: np.ndarray) -> np.ndarray:
    band = np.zeros((34, array.shape[1], 3), dtype=np.uint8)
    band[:] = 24
    cv2.putText(band, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, array])


def _hstack(panels: list[np.ndarray], gap: int = 12) -> np.ndarray:
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for index, panel in enumerate(panels):
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
        if index != len(panels) - 1:
            padded.append(np.full((height, gap, 3), 45, dtype=np.uint8))
    return np.hstack(padded)


def _class_scene(image_rgb: np.ndarray, labels: np.ndarray, class_ids: tuple[int, ...]) -> np.ndarray:
    """One parsed class tinted on the dimmed photo -- real segmentation evidence."""
    mask = np.zeros(labels.shape, np.uint8)
    for class_id in class_ids:
        mask[labels == class_id] = 255
    colour = CLASS_COLOURS.get(class_ids[0], DEFAULT_CLASS_COLOUR)
    return _tint(_dim(image_rgb, mask), mask, colour, 0.62)


def _hairline_crop(image_rgb, mask_set, boundaries: dict[str, float]) -> np.ndarray:
    """Zoom on the hairline / new-hairstyle junction, plus the growth rim."""
    x0, y0, x1, y1 = (int(round(value)) for value in mask_set.stats["faceBox"])
    face_w = float(mask_set.stats["faceWidthPx"])
    face_h = float(mask_set.stats["faceHeightPx"])
    left = max(0, int(x0 - 0.55 * face_w))
    right = min(image_rgb.shape[1], int(x1 + 0.55 * face_w))
    top = max(0, int(y0 - 0.45 * face_h))
    bottom = min(image_rgb.shape[0], int(y0 + 0.75 * face_h))

    hair = mask_set.hair
    protected = mask_set.protected
    parsed_hair = np.zeros(labels_shape(mask_set), np.uint8)
    parsed_hair[mask_set.label_map == 2] = 255
    rim = cv2.bitwise_and(hair, cv2.bitwise_not(parsed_hair))
    contact = cv2.bitwise_and(dilate_mask(hair, 2.0), protected)

    crop = _dim(image_rgb, np.maximum(hair, protected), alpha=0.25)
    crop = _tint(crop, hair, (40, 200, 255), 0.45)
    crop = _tint(crop, rim, (0, 255, 120), 0.55)
    crop = _tint(crop, protected, (255, 40, 40), 0.40)
    crop = _tint(crop, contact, (255, 255, 255), 0.85)
    panel = crop[top:bottom, left:right]
    panel = cv2.resize(panel, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    return _label(
        "hairline junction (2x): cyan=parsed hair, grown as the edit region  "
        "green=growth rim outside the old hair  red=protected face/neck  "
        "white=hairline contact",
        panel,
    )


def labels_shape(mask_set) -> tuple[int, int]:
    return mask_set.label_map.shape


def _acceptance(image_rgb, labels, landmarks, mask_set) -> dict:
    stats = mask_set.stats
    protected, editable, hair, clothing = (
        mask_set.protected,
        mask_set.editable,
        mask_set.hair,
        mask_set.clothing,
    )
    parsed_face = mask_set.parsed_face
    parsed_hair = np.zeros(labels.shape, np.uint8)
    parsed_hair[labels == 2] = 255
    rim = cv2.bitwise_and(hair, cv2.bitwise_not(parsed_hair))

    # Every check below is recomputed from the masks themselves rather than
    # read back from ``stats``: a stale or wrong stat must not be able to make
    # the gate pass. This is the "do not silently claim success" property.
    overlap_pixels = int((cv2.bitwise_and(protected, editable) > 0).sum())
    background_label = np.zeros(labels.shape, np.uint8)
    background_label[labels == 0] = 255
    editable_pixels = int((editable > 0).sum())
    background_editable = int(
        (cv2.bitwise_and(background_label, editable) > 0).sum()
    )
    background_share = background_editable / float(max(editable_pixels, 1))

    feature_hits = []
    for group in FEATURE_GROUPS + (LIPS_INNER,):
        for index in group:
            x, y = landmarks[index]
            feature_hits.append(int(protected[int(round(y)), int(round(x))]) > 0)
    glasses_hits = [
        int(protected[int(round(landmarks[index][1])), int(round(landmarks[index][0]))]) > 0
        for index in GLASSES_BAND
    ]

    # The *face* is the anatomy, not everything the parser filed as Face: a
    # white high collar below the chin arrives as class 11 and is deliberately
    # editable, so the garment check is scored above the chin row only. The row
    # is the one the mask builder uses -- the chin landmark plus its margin --
    # because the few rows just under the landmark are still jaw, and the mask
    # keeps them by design.
    face_w = face_width(landmarks)
    chin_row = int(
        round(
            float(landmarks[CHIN][1])
            + SemanticMaskConfig().collar_margin_ratio * face_w
        )
    )
    above_chin = np.zeros(labels.shape, np.uint8)
    above_chin[:chin_row, :] = 255
    face_above_chin = cv2.bitwise_and(parsed_face, above_chin)
    clothing_on_face = int((cv2.bitwise_and(clothing, face_above_chin) > 0).sum())

    # The old-collar gate. Recomputed from the photo and the landmarks, never
    # read back from ``stats``: the pixels the parser filed as Face below the
    # chin that are *not* the subject's own skin are the garment that used to be
    # frozen with the face. They must now be inside the edit area, and the
    # throat (below-chin skin) must still be kept.
    reference, _ = skin_reference_colour(image_rgb, landmarks, face_w)
    below_chin_face = cv2.bitwise_and(parsed_face, cv2.bitwise_not(above_chin))
    below_chin_pixels = int((below_chin_face > 0).sum())
    collar_pixels = 0
    collar_editable = 0
    neck_skin_protected = 0
    collar_colour: list[float] = []
    collar_separable = reference is not None
    if collar_separable and below_chin_pixels > 0:
        skin = adaptive_skin_mask(image_rgb, reference, face_w, FaceLockConfig())
        collar = cv2.bitwise_and(below_chin_face, cv2.bitwise_not(skin))
        neck = cv2.bitwise_and(below_chin_face, skin)
        collar_pixels = int((collar > 0).sum())
        collar_editable = int((cv2.bitwise_and(collar, editable) > 0).sum())
        neck_skin_protected = int((cv2.bitwise_and(neck, protected) > 0).sum())
        if collar_pixels:
            collar_colour = [
                round(float(value), 1)
                for value in image_rgb[collar > 0].mean(axis=0)
            ]

    distinct = len(np.unique(labels))

    checks = {
        "hairIsNotALegacyCircle": bool(
            stats["hairIoUWithLegacyEllipse"] < 0.95
            and stats["hairOutsideLegacyEllipseShare"] > 0.05
        ),
        "backgroundStaysOutOfEditArea": bool(background_share < 0.35),
        "protectedAndEditableAreDisjoint": bool(overlap_pixels == 0),
        "faceFeaturesAreInsideProtected": bool(all(feature_hits)),
        "glassesBandIsInsideProtected": bool(all(glasses_hits)),
        "clothingDoesNotCoverTheFace": bool(clothing_on_face == 0),
        "staleCollarEntersEditArea": bool(
            below_chin_pixels == 0
            or (collar_separable and collar_editable >= collar_pixels)
        ),
        "newHairstyleMayExceedOldHair": bool(int((rim > 0).sum()) > 0),
        "labelMapIsRealSegmentation": bool(distinct >= 4 and stats["hairPixels"] > 0),
    }
    return {
        "checks": checks,
        "allPassed": bool(all(checks.values())),
        "evidence": {
            "hairIoUWithLegacyEllipse": stats["hairIoUWithLegacyEllipse"],
            "hairOutsideLegacyEllipseShare": stats["hairOutsideLegacyEllipseShare"],
            "backgroundEditablePixels": background_editable,
            "backgroundEditableShare": round(background_share, 4),
            "protectedEditableOverlapPixels": overlap_pixels,
            "parsedFacePixels": stats["parsedFacePixels"],
            "parsedFaceIslandPixels": stats["parsedFaceIslandPixels"],
            "faceSkinShare": stats["faceSkinShare"],
            "clothingPixelsOnParsedFace": clothing_on_face,
            "parsedFaceBelowChinPixels": below_chin_pixels,
            "collarPixelsBelowTheChin": collar_pixels,
            "collarPixelsInsideEditArea": collar_editable,
            "neckSkinPixelsStillProtected": neck_skin_protected,
            "collarSeparableFromNeck": bool(collar_separable),
            "collarMeanColourRgb": collar_colour,
            "collarSplitApplied": bool(stats.get("collarSplitApplied", False)),
            "faceCorePixels": int(stats.get("faceCorePixels", 0)),
            "newSilhouetteRimPixels": int((rim > 0).sum()),
            "featureLandmarkCount": len(feature_hits),
            "glassesLandmarkCount": len(glasses_hits),
            "distinctParsedClasses": int(distinct),
            "hairContactWithProtectedPixels": stats["hairContactWithProtectedPixels"],
            "speckleIslandsDropped": int(
                stats["hairIslandsDropped"] + stats["clothingIslandsDropped"]
            ),
            "specklePixelsDropped": stats["specklePixelsDropped"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = Path(args.image)
    labels_path = Path(args.labels)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path} (no download is attempted).")
    if not labels_path.is_file():
        raise SystemExit(
            f"Label map not found: {labels_path}. Run tools.run_human_parsing in the "
            "environment that has onnxruntime first; this tool will not draw a "
            "geometric stand-in in its place."
        )
    if not args.landmarks_json and not Path(args.model).is_file():
        raise SystemExit(
            f"Face landmark model not found: {args.model}. "
            "Run tools.download_face_landmarker once, or pass --landmarks-json."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    labels = np.asarray(Image.open(labels_path))
    if labels.ndim != 2:
        raise SystemExit("The label map must be a single-channel PNG of ATR class ids.")
    if labels.shape != image_rgb.shape[:2]:
        raise SystemExit(
            f"Label map is {labels.shape}, the photo is {image_rgb.shape[:2]}. The "
            "mask would not be in the photo's own coordinates; refusing to continue."
        )

    landmarks = _load_landmarks(args, image_rgb)
    mask_set = build_semantic_edit_masks(image_rgb, landmarks, labels)
    protected, editable = mask_set.protected, mask_set.editable

    # --- the regions the brief asks for, all in photo coordinates -----------
    _save(out_dir / "p27_original.png", image_rgb)
    _save(out_dir / "p27_mask_hair.png", mask_set.hair)
    _save(out_dir / "p27_mask_clothing.png", mask_set.clothing)
    _save(out_dir / "p27_mask_combined.png", editable)
    _save(out_dir / "p27_mask_face_protected.png", protected)
    _save(out_dir / "p27_mask_background.png", mask_set.background_preserve)
    _save(out_dir / "p27_inpaint_mask.png", editable)

    colour = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, name in ATR_CLASSES.items():
        colour[labels == class_id] = CLASS_COLOURS.get(class_id, DEFAULT_CLASS_COLOUR)
    _save(out_dir / "p27_segmentation.png", colour)

    for class_id, name, ids in (
        ("hair", "Hair", (2,)),
        ("face", "Face", (11,)),
        ("clothing", "Upper-clothes", (4,)),
        ("background", "Background", (0,)),
    ):
        _save(out_dir / f"p27_class_{class_id}.png", _class_scene(image_rgb, labels, ids))

    overlay = _dim(image_rgb, np.maximum(protected, editable), alpha=0.35)
    overlay = _tint(overlay, protected, (255, 40, 40), 0.45)
    overlay = _tint(overlay, editable, (40, 220, 80), 0.45)
    _save(out_dir / "p27_overlay.png", overlay)

    _save(out_dir / "p27_hairline_junction.png", _hairline_crop(image_rgb, mask_set, {}))

    # --- P2.6 vs P2.7, side by side ----------------------------------------
    legacy = build_edit_masks(image_rgb, landmarks)
    legacy_protected = _tint(_dim(image_rgb, legacy.protected), legacy.protected, (255, 40, 40), 0.45)
    legacy_editable = _tint(_dim(image_rgb, legacy.editable), legacy.editable, (40, 220, 80), 0.45)
    semantic_protected = _tint(_dim(image_rgb, protected), protected, (255, 40, 40), 0.45)
    semantic_editable = _tint(_dim(image_rgb, editable), editable, (40, 220, 80), 0.45)
    comparison = _hstack(
        [
            _label("P2.6 geometric protected (red)", legacy_protected),
            _label("P2.6 geometric editable (green)", legacy_editable),
            _label("P2.7 parsed protected (red)", semantic_protected),
            _label("P2.7 parsed editable (green)", semantic_editable),
        ]
    )
    _save(out_dir / "p27_vs_p26.png", comparison)

    acceptance = _acceptance(image_rgb, labels, landmarks, mask_set)
    stats = dict(mask_set.stats)
    stats["legacy"] = {
        "protectedPixels": legacy.stats["protectedPixels"],
        "hairPixels": legacy.stats["hairPixels"],
        "clothingPixels": legacy.stats["clothingPixels"],
        "editablePixels": legacy.stats["editablePixels"],
        "editableShare": legacy.stats["editableShare"],
    }

    record = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "command": "python -m tools.build_parsing_masks",
        "image": str(image_path),
        "labelMap": str(labels_path),
        "model": str(args.model) if not args.landmarks_json else "landmarks-json",
        "paidGenerations": 0,
        "paidVisionCalls": 0,
        "note": (
            "Masks built from a real ATR human-parsing inference result. The hair "
            "and clothing regions are parsed silhouettes, not geometric stand-ins. "
            "No generator was run."
        ),
        "stats": stats,
        "gaps": list(mask_set.gaps),
        "acceptance": acceptance,
        "outputs": sorted(path.name for path in out_dir.glob("p27_*.png")),
    }
    stats_path = out_dir / "p27-parsing-mask-stats.json"
    stats_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote the P2.7 PNGs + {stats_path.name} to {out_dir}")
    print(
        "  protected {protectedShare:.1%} | editable {editableShare:.1%} | "
        "hair {hairPixels} | clothing {clothingPixels}".format(**stats)
    )
    print(
        "  hair vs P2.6 ellipse: IoU {hairIoUWithLegacyEllipse} | outside "
        "{hairOutsideLegacyEllipseShare:.1%}".format(**stats)
    )
    print(
        "  background in edit area: {backgroundEditablePixels} px "
        "({backgroundEditableShare:.1%} of the edit area)".format(**stats)
    )
    print(
        "  face island {parsedFaceIslandPixels} px | skin-consistent "
        "{faceSkinShare:.1%} | speckle dropped {specklePixelsDropped} px "
        "({hairIslandsDropped}+{clothingIslandsDropped} islands)".format(**stats)
    )
    for name, passed in acceptance["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    for gap in mask_set.gaps:
        print(f"  gap: {gap}")
    print("No generation, no paid provider, no download was performed.")
    return 0 if acceptance["allPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
