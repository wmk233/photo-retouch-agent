"""Local edit mask: geometry, semantics and the offline preview tool."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.domains.portrait.edit_mask import EditMaskConfig, build_edit_masks
from app.domains.portrait.face_lock import (
    BROW_A,
    BROW_B,
    CHIN,
    EYE_A,
    EYE_B,
    FACE_OVAL,
    FACE_SIDES,
    LIPS_INNER,
    NOSE,
)
from tools import build_edit_masks as tool

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 360
FACE_CX = 180.0
FACE_CY = 200.0
FACE_RX = 60.0
FACE_RY = 80.0
SKIN = np.array([200, 170, 150], dtype=np.uint8)


def _ring(points: np.ndarray, indices, center, rx: float, ry: float) -> None:
    count = len(indices)
    for position, index in enumerate(indices):
        angle = 2.0 * np.pi * position / count
        points[index] = [
            center[0] + rx * np.cos(angle),
            center[1] + ry * np.sin(angle),
        ]


def _synthetic_landmarks() -> np.ndarray:
    """A 478-point mesh shaped like a simple face -- no photo, no model."""
    points = np.tile([FACE_CX, FACE_CY], (478, 1)).astype(np.float64)
    _ring(points, FACE_OVAL, (FACE_CX, FACE_CY), FACE_RX, FACE_RY)
    points[FACE_SIDES[0]] = [FACE_CX - FACE_RX, FACE_CY]
    points[FACE_SIDES[1]] = [FACE_CX + FACE_RX, FACE_CY]
    _ring(points, EYE_A, (FACE_CX - 26.0, FACE_CY - 22.0), 14.0, 7.0)
    _ring(points, EYE_B, (FACE_CX + 26.0, FACE_CY - 22.0), 14.0, 7.0)
    _ring(points, BROW_A, (FACE_CX - 26.0, FACE_CY - 34.0), 15.0, 4.0)
    _ring(points, BROW_B, (FACE_CX + 26.0, FACE_CY - 34.0), 15.0, 4.0)
    _ring(points, NOSE, (FACE_CX, FACE_CY + 8.0), 8.0, 14.0)
    _ring(points, LIPS_INNER, (FACE_CX, FACE_CY + 44.0), 12.0, 6.0)
    points[CHIN] = [FACE_CX, FACE_CY + FACE_RY]
    return points


def _image() -> np.ndarray:
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:, :] = SKIN
    return image


@pytest.fixture()
def mask_set():
    return build_edit_masks(_image(), _synthetic_landmarks())


def test_editable_and_protected_are_disjoint(mask_set):
    overlap = cv2.bitwise_and(mask_set.editable, mask_set.protected)
    assert int(overlap.sum()) == 0
    assert mask_set.stats["identityCoversEditable"] is True


def test_preserve_is_the_exact_complement_of_editable(mask_set):
    assert np.array_equal(mask_set.preserve, cv2.bitwise_not(mask_set.editable))


def test_unselected_pixels_are_kept(mask_set):
    """preserve must cover every pixel the generator was not told to repaint."""
    editable = mask_set.editable
    keep = mask_set.preserve
    assert int(editable[keep > 0].sum()) == 0
    assert int(keep[editable > 0].sum()) == 0


def test_identity_features_are_protected(mask_set):
    landmarks = _synthetic_landmarks()
    protected = mask_set.protected
    for group in (EYE_A, EYE_B, BROW_A, BROW_B, NOSE, LIPS_INNER):
        for index in group:
            x, y = landmarks[index]
            assert protected[int(round(y)), int(round(x))] > 0, index


def test_teeth_inside_the_smile_are_protected(mask_set):
    landmarks = _synthetic_landmarks()
    inner = landmarks[list(LIPS_INNER)]
    centre = inner.mean(axis=0)
    assert mask_set.protected[int(round(centre[1])), int(round(centre[0]))] > 0


def test_hair_and_clothing_bands_are_open(mask_set):
    assert mask_set.stats["hairPixels"] > 0
    assert mask_set.stats["clothingPixels"] > 0
    assert mask_set.editable.max() == 255
    # A point above the forehead and a point below the chin are repaintable.
    assert mask_set.editable[100, int(FACE_CX)] == 255
    assert mask_set.editable[IMAGE_HEIGHT - 5, int(FACE_CX)] == 255


def test_feather_weights_are_normalised(mask_set):
    for weight in (mask_set.editable_weight(), mask_set.protected_weight()):
        assert weight.dtype == np.float32
        assert 0.0 <= float(weight.min())
        assert float(weight.max()) <= 1.0
    assert mask_set.feather_px > 0


def test_gaps_state_the_bands_are_heuristic(mask_set):
    joined = " ".join(mask_set.gaps)
    assert "hair_band_is_geometric" in joined
    assert "clothing_band_is_geometric" in joined


def test_rejects_non_photo_input():
    with pytest.raises(ValueError):
        build_edit_masks(np.zeros((10, 10), dtype=np.uint8), _synthetic_landmarks())


def test_rejects_degenerate_landmarks():
    flat = np.zeros((478, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        build_edit_masks(_image(), flat)


def test_region_below_floor_is_dropped_and_reported():
    config = EditMaskConfig(min_region_px=10 ** 9)
    result = build_edit_masks(_image(), _synthetic_landmarks(), config=config)
    assert result.stats["hairPixels"] == 0
    assert result.stats["clothingPixels"] == 0
    joined = " ".join(result.gaps)
    assert "hair_region_below_floor" in joined
    assert "clothing_region_below_floor" in joined


def test_preview_tool_writes_masks_without_any_model(tmp_path: Path):
    image_path = tmp_path / "portrait.png"
    from PIL import Image

    Image.fromarray(_image()).save(image_path)
    landmarks_path = tmp_path / "landmarks.json"
    landmarks_path.write_text(
        json.dumps(_synthetic_landmarks().tolist()), encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    exit_code = tool.main(
        [
            "--image",
            str(image_path),
            "--landmarks-json",
            str(landmarks_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    for name in ("p26_original.png", "p26_protected.png", "p26_editable.png", "p26_overlay.png"):
        assert (out_dir / name).is_file()
    record = json.loads((out_dir / "p26-edit-mask-stats.json").read_text(encoding="utf-8"))
    assert record["paidGenerations"] == 0
    assert record["paidVisionCalls"] == 0
    assert record["stats"]["editablePixels"] > 0
    assert record["gaps"]
