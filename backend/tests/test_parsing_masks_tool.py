"""``tools.build_parsing_masks``: it must refuse a stand-in, and only ship real masks.

The tool is the bridge between the parser's label map and the reviewable PNGs.
Its whole contract is that the label map is *required* and must already be in the
photo's own coordinates -- if it is not, the tool stops instead of drawing the
P2.6 ellipse and reporting success.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
from app.domains.portrait.human_parsing import (
    BACKGROUND,
    FACE,
    HAIR,
    UPPER_CLOTHES,
)
from tools import build_parsing_masks as tool

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 360
FACE_CX = 180.0
FACE_CY = 200.0
FACE_RX = 60.0
FACE_RY = 80.0
SKIN = np.array([200, 170, 150], dtype=np.uint8)

EXPECTED_OUTPUTS = (
    "p27_original.png",
    "p27_mask_hair.png",
    "p27_mask_clothing.png",
    "p27_mask_combined.png",
    "p27_mask_face_protected.png",
    "p27_mask_background.png",
    "p27_inpaint_mask.png",
    "p27_segmentation.png",
    "p27_overlay.png",
    "p27_hairline_junction.png",
    "p27_vs_p26.png",
)


def _ring(points, indices, center, rx, ry) -> None:
    count = len(indices)
    for position, index in enumerate(indices):
        angle = 2.0 * np.pi * position / count
        points[index] = [
            center[0] + rx * np.cos(angle),
            center[1] + ry * np.sin(angle),
        ]


def _landmarks() -> np.ndarray:
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


def _labels() -> np.ndarray:
    labels = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    labels[40:118, 110:250] = HAIR
    labels[120:280, 120:240] = FACE
    labels[300:470, 70:290] = UPPER_CLOTHES
    return labels


@pytest.fixture()
def review_scene(tmp_path: Path) -> dict[str, Path]:
    image_path = tmp_path / "portrait.png"
    Image.fromarray(np.broadcast_to(SKIN, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).copy()).save(
        image_path
    )
    labels_path = tmp_path / "p27_labels.png"
    Image.fromarray(_labels()).save(labels_path)
    landmarks_path = tmp_path / "landmarks.json"
    landmarks_path.write_text(json.dumps(_landmarks().tolist()), encoding="utf-8")
    return {
        "image": image_path,
        "labels": labels_path,
        "landmarks": landmarks_path,
        "out": tmp_path / "out",
    }


def test_tool_stops_without_a_label_map(tmp_path: Path):
    image_path = tmp_path / "portrait.png"
    Image.fromarray(np.broadcast_to(SKIN, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).copy()).save(
        image_path
    )
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="Label map not found"):
        tool.main(
            [
                "--image",
                str(image_path),
                "--labels",
                str(tmp_path / "absent.png"),
                "--out-dir",
                str(out_dir),
            ]
        )
    if out_dir.is_dir():
        assert not list(out_dir.glob("p27_*.png"))


def test_tool_rejects_a_label_map_in_the_wrong_coordinates(review_scene):
    bad_labels = review_scene["labels"].parent / "bad_labels.png"
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(bad_labels)
    with pytest.raises(SystemExit, match="photo's own coordinates"):
        tool.main(
            [
                "--image",
                str(review_scene["image"]),
                "--labels",
                str(bad_labels),
                "--landmarks-json",
                str(review_scene["landmarks"]),
                "--out-dir",
                str(review_scene["out"]),
            ]
        )


def test_tool_writes_every_mask_from_a_real_label_map(review_scene):
    exit_code = tool.main(
        [
            "--image",
            str(review_scene["image"]),
            "--labels",
            str(review_scene["labels"]),
            "--landmarks-json",
            str(review_scene["landmarks"]),
            "--out-dir",
            str(review_scene["out"]),
        ]
    )
    out_dir = review_scene["out"]
    for name in EXPECTED_OUTPUTS:
        assert (out_dir / name).is_file(), name

    record = json.loads(
        (out_dir / "p27-parsing-mask-stats.json").read_text(encoding="utf-8")
    )
    assert record["paidGenerations"] == 0
    assert record["paidVisionCalls"] == 0
    assert record["labelMap"] == str(review_scene["labels"])
    assert exit_code == (0 if record["acceptance"]["allPassed"] else 1)
    assert set(record["acceptance"]["checks"]) == {
        "hairIsNotALegacyCircle",
        "backgroundStaysOutOfEditArea",
        "protectedAndEditableAreDisjoint",
        "faceFeaturesAreInsideProtected",
        "glassesBandIsInsideProtected",
        "clothingDoesNotCoverTheFace",
        "staleCollarEntersEditArea",
        "newHairstyleMayExceedOldHair",
        "labelMapIsRealSegmentation",
    }
    assert record["stats"]["semantic"] is True
    assert record["gaps"]


def test_inpaint_mask_is_binary_and_matches_the_combined_mask(review_scene):
    tool.main(
        [
            "--image",
            str(review_scene["image"]),
            "--labels",
            str(review_scene["labels"]),
            "--landmarks-json",
            str(review_scene["landmarks"]),
            "--out-dir",
            str(review_scene["out"]),
        ]
    )
    out_dir = review_scene["out"]
    combined = np.asarray(Image.open(out_dir / "p27_mask_combined.png"))
    inpaint = np.asarray(Image.open(out_dir / "p27_inpaint_mask.png"))
    assert set(np.unique(inpaint).tolist()) <= {0, 255}
    assert np.array_equal(combined, inpaint)
    # The protected face mask and the edit mask never share a white pixel.
    protected = np.asarray(Image.open(out_dir / "p27_mask_face_protected.png"))
    assert not np.any((protected > 0) & (inpaint > 0))
