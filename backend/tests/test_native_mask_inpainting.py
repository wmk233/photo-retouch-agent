"""The mask layer refuses honestly: no parser, no face, no trustworthy mask.

``NativeMaskInpainting`` is the only place that decides whether a photo may be
handed to a *paid* masked backend. The brief is explicit that it must fail --
loudly, with a recorded reason -- rather than ship a mask it cannot stand behind
(an unseparated old collar, an empty edit area, background flooding the frame).

Every test injects a stub detector/segmenter at the designed seam, so no
onnxruntime, no network and no paid call is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.domains.portrait.edit_mask import SemanticMaskConfig
from app.domains.portrait.inpaint_verify import InpaintVerificationError
from app.services.native_mask_inpainting import (
    MAX_BACKGROUND_SHARE_OF_EDIT_AREA,
    NativeMaskInpainting,
    NativeMaskUnavailable,
)
from tests.test_semantic_edit_mask import _image, _labels, _landmarks


class StubDetector:
    def __init__(self, faces: list[np.ndarray] | None = None) -> None:
        self.faces = faces

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        if self.faces is None:
            return [_landmarks()]
        return self.faces

    def close(self) -> None:
        return None


class StubSegmenter:
    def segment(self, image: np.ndarray) -> np.ndarray:
        return _labels()


def _settings(tmp_path: Path, **overrides) -> Settings:
    models = tmp_path / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    landmark = models / "face_landmarker.task"
    landmark.write_bytes(b"stub")
    parsing = models / "schp-atr-18.onnx"
    parsing.write_bytes(b"stub")
    values = {
        "data_dir": tmp_path / "data",
        "face_landmark_model_path": landmark,
        "human_parsing_model_path": parsing,
        "face_lock_mode": "off",
    }
    values.update(overrides)
    return Settings(**values)


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "portrait.png"
    Image.fromarray(_image()).save(path)
    return path


def _native(
    settings: Settings,
    *,
    detector: StubDetector | None = None,
    segmenter: StubSegmenter | None = None,
    mask_config: SemanticMaskConfig | None = None,
) -> NativeMaskInpainting:
    return NativeMaskInpainting(
        settings,
        detector_factory=lambda path: detector or StubDetector(),
        segmenter_factory=lambda path: segmenter or StubSegmenter(),
        mask_config=mask_config,
    )


# ------------------------------------------------------------------ build_mask


def test_build_mask_writes_a_black_white_mask_and_its_provenance(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    mask_path = settings.outputs_dir / "mask_test.png"

    build = _native(settings).build_mask(source, mask_path)

    assert mask_path.exists()
    mask = np.asarray(Image.open(mask_path).convert("L"))
    photo = np.asarray(Image.open(source).convert("RGB"))
    assert mask.shape == photo.shape[:2], "the mask is the photo's own size"
    assert set(np.unique(mask)) <= {0, 255}, "strictly black/white, no grey"
    assert 0 < int((mask == 255).sum()) < mask.size

    assert build.acceptance["passed"] is True
    assert build.stats["identityCoversEditable"] is True
    assert build.stats["protectedCoversEditable"] is True

    record = json.loads(
        (settings.outputs_dir / "_meta" / "mask_test.png.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["kind"] == "native-edit-mask"
    assert record["semantics"] == "white=repaint black=keep original pixels"
    assert record["acceptance"]["passed"] is True
    assert "no photo left the machine" in record["note"]


def test_build_mask_refuses_when_the_parser_model_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, human_parsing_model_path=tmp_path / "missing.onnx")
    source = _source(tmp_path)
    mask_path = settings.outputs_dir / "mask_test.png"

    with pytest.raises(NativeMaskUnavailable, match="human-parsing model is missing"):
        _native(settings).build_mask(source, mask_path)

    assert not mask_path.exists(), "an unbuildable mask is never written"
    assert not (settings.outputs_dir / "_meta" / "mask_test.png.json").exists()


def test_build_mask_refuses_when_the_landmark_model_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, face_landmark_model_path=tmp_path / "missing.task")
    source = _source(tmp_path)

    with pytest.raises(NativeMaskUnavailable, match="face landmark model is missing"):
        _native(settings).build_mask(source, settings.outputs_dir / "mask.png")


def test_build_mask_refuses_when_no_face_is_detected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)

    with pytest.raises(NativeMaskUnavailable, match="No face was detected"):
        _native(settings, detector=StubDetector(faces=[])).build_mask(
            source, settings.outputs_dir / "mask.png"
        )


def test_build_mask_refuses_when_the_collar_cannot_be_separated(tmp_path: Path) -> None:
    """An unsampleable skin tone leaves a blocking gap; the mask is not shipped."""
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    tiny = np.array([[180.0, 200.0]]) + (_landmarks() - [180.0, 200.0]) * 0.05

    with pytest.raises(NativeMaskUnavailable, match="collar_split_unresolved"):
        _native(settings, detector=StubDetector(faces=[tiny])).build_mask(
            source, settings.outputs_dir / "mask.png"
        )
    assert not (settings.outputs_dir / "mask.png").exists()


# ------------------------------------------------------------------- acceptance


def test_acceptance_refuses_background_flooding_the_edit_area(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    mask_path = settings.outputs_dir / "mask.png"
    flooding = SemanticMaskConfig(hair_grow_ratio=0.2, clothing_grow_ratio=0.2)

    with pytest.raises(NativeMaskUnavailable, match="background makes up"):
        _native(settings, mask_config=flooding).build_mask(source, mask_path)

    assert not mask_path.exists(), "a mask that opens the room is never shipped"


def test_background_limit_is_a_quarter_of_the_edit_area() -> None:
    assert MAX_BACKGROUND_SHARE_OF_EDIT_AREA == 0.25


def test_acceptance_refuses_an_empty_edit_area(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    empty = SemanticMaskConfig(min_region_px=10**9)

    with pytest.raises(NativeMaskUnavailable, match="nothing meaningful"):
        _native(settings, mask_config=empty).build_mask(
            source, settings.outputs_dir / "mask.png"
        )


# --------------------------------------------------------------------- finalise


def test_finalise_composites_kept_pixels_from_the_original(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    mask_path = settings.outputs_dir / "mask.png"
    build = _native(settings).build_mask(source, mask_path)

    original = np.asarray(Image.open(source).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    painted = np.zeros_like(original)
    painted[:, :] = (18, 24, 48)
    generated = np.where((mask == 255)[..., None], painted, original)
    raw_path = settings.outputs_dir / "raw.png"
    Image.fromarray(generated.astype(np.uint8)).save(raw_path)
    final_path = settings.outputs_dir / "final.png"

    verification = _native(settings).finalise(
        source, raw_path, mask_path, final_path, mask_set=build.mask_set
    )

    assert verification.ok is True
    assert verification.statuses["MASK_VALID"] is True
    assert verification.statuses["PIXEL_IDENTITY_PASSED"] is True
    assert verification.statuses["VISUAL_ACCEPTANCE_PENDING"] is True
    final = np.asarray(Image.open(final_path).convert("RGB"))
    kept = mask == 0
    assert np.array_equal(final[kept], original[kept])
    assert np.array_equal(final[mask == 255], painted[mask == 255])


def test_finalise_refuses_an_output_it_cannot_register(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _source(tmp_path)
    mask_path = settings.outputs_dir / "mask.png"
    build = _native(settings).build_mask(source, mask_path)

    resized = settings.outputs_dir / "raw.png"
    photo = Image.open(source).convert("RGB")
    photo.resize((photo.width * 2, photo.height * 2)).save(resized)
    final_path = settings.outputs_dir / "final.png"

    with pytest.raises(InpaintVerificationError, match="does not match the original"):
        _native(settings).finalise(
            source, resized, mask_path, final_path, mask_set=build.mask_set
        )

    assert not final_path.exists(), "an unregisterable result is never published"
