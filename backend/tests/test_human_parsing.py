"""The ATR human parser: preprocessing, argmax, and the no-fallback contract.

onnxruntime is deliberately absent from this environment's dependencies, so
the runtime tests here assert the *failure* path: the segmenter must raise
``HumanParsingUnavailable`` and must never hand back a stand-in label map. The
pre/post-processing is pure numpy/PIL and is tested against the exact values
the network expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.domains.portrait import human_parsing as hp
from tools import run_human_parsing


# ------------------------------------------------------------------ taxonomy


def test_atr_taxonomy_ids_match_the_network_channel_order():
    assert hp.CLASS_COUNT == 18
    assert hp.BACKGROUND == 0
    assert hp.HAIR == 2
    assert hp.SUNGLASSES == 3
    assert hp.UPPER_CLOTHES == 4
    assert hp.FACE == 11
    assert hp.ATR_CLASSES[hp.HAIR] == "Hair"
    assert hp.ATR_CLASSES[hp.FACE] == "Face"


def test_face_and_clothing_groups_are_the_right_classes():
    # ATR has no neck class, so the neck arrives as Face.
    assert hp.FACE_CLASSES == (hp.FACE,)
    # Hats, sunglasses and scarves sit on the body; they are not garments.
    assert hp.HAT not in hp.CLOTHING_CLASSES
    assert hp.SUNGLASSES not in hp.CLOTHING_CLASSES
    assert hp.SCARF not in hp.CLOTHING_CLASSES
    assert hp.UPPER_CLOTHES in hp.CLOTHING_CLASSES


def test_default_model_path_points_at_the_pinned_export():
    path = hp.default_model_path(Path("/models"))
    assert path == Path("/models") / hp.HUMAN_PARSING_MODEL_FILENAME
    assert path.name == "schp-atr-18.onnx"


def test_provenance_is_pinned_under_mit():
    assert hp.HUMAN_PARSING_LICENSE == "MIT"
    assert "Self-Correction-Human-Parsing" in hp.HUMAN_PARSING_UPSTREAM
    assert len(hp.HUMAN_PARSING_SHA256) == 64
    assert len(hp.HUMAN_PARSING_DATA_SHA256) == 64
    assert hp.INPUT_SIZE == (512, 512)


# ------------------------------------------------------------ pre-processing


def test_preprocess_returns_the_normalised_chw_tensor():
    image = np.full((64, 48, 3), 128, dtype=np.uint8)
    tensor = hp.preprocess(image)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32

    scaled = 128.0 / 255.0
    expected = [
        (scaled - mean) / std
        for mean, std in zip(hp.PIXEL_MEAN, hp.PIXEL_STD)
    ]
    for channel, value in enumerate(expected):
        assert np.allclose(tensor[0, channel], value, atol=1e-4)


def test_preprocess_rejects_a_non_rgb_image():
    with pytest.raises(ValueError, match="HxWx3"):
        hp.preprocess(np.zeros((64, 48), dtype=np.uint8))


# ----------------------------------------------------------- post-processing


def test_label_map_is_the_argmax_at_the_target_size():
    logits = np.zeros((1, hp.CLASS_COUNT, 4, 4), dtype=np.float32)
    logits[0, hp.UPPER_CLOTHES] = 5.0
    labels = hp.label_map_from_logits(logits, (8, 8))
    assert labels.shape == (8, 8)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels).tolist()) == {hp.UPPER_CLOTHES}


def test_label_map_upscale_never_blends_two_classes():
    logits = np.zeros((1, hp.CLASS_COUNT, 2, 2), dtype=np.float32)
    logits[0, hp.HAIR, :, :1] = 1.0
    logits[0, hp.UPPER_CLOTHES, :, 1:] = 1.0
    labels = hp.label_map_from_logits(logits, (8, 8))
    assert set(np.unique(labels).tolist()) <= {hp.HAIR, hp.UPPER_CLOTHES}


def test_label_map_rejects_the_wrong_channel_count():
    with pytest.raises(ValueError, match="ATR classes"):
        hp.label_map_from_logits(np.zeros((1, 17, 4, 4), np.float32), (4, 4))


def test_label_map_rejects_a_non_4d_tensor():
    with pytest.raises(ValueError, match="N, C, H, W"):
        hp.label_map_from_logits(np.zeros((18, 4, 4), np.float32), (4, 4))


# ------------------------------------------------------------------- runtime


def test_segmenter_reports_a_missing_weight(tmp_path: Path):
    segmenter = hp.SchpOnnxSegmenter(tmp_path / "absent.onnx")
    with pytest.raises(hp.HumanParsingUnavailable, match="not found"):
        segmenter.segment(np.zeros((16, 16, 3), dtype=np.uint8))


def test_segmenter_never_returns_a_stand_in_mask(tmp_path: Path):
    """A weight that cannot be loaded raises; it does not fabricate a label map."""
    fake = tmp_path / "schp-atr-18.onnx"
    fake.write_bytes(b"not a real onnx graph")
    segmenter = hp.SchpOnnxSegmenter(fake)
    with pytest.raises(hp.HumanParsingUnavailable) as excinfo:
        segmenter.segment(np.zeros((16, 16, 3), dtype=np.uint8))
    assert "onnxruntime" in str(excinfo.value)


def test_segmenter_rejects_a_non_rgb_image(tmp_path: Path):
    segmenter = hp.SchpOnnxSegmenter(tmp_path / "absent.onnx")
    with pytest.raises(ValueError, match="HxWx3"):
        segmenter.segment(np.zeros((16, 16), dtype=np.uint8))


# -------------------------------------------------------------- the CLI tool


def _write_image(path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((32, 24, 3), dtype=np.uint8)).save(path)


def test_cli_stops_when_the_photo_is_missing(tmp_path: Path):
    with pytest.raises(SystemExit, match="Image not found"):
        run_human_parsing.main(["--image", str(tmp_path / "absent.jpg")])


def test_cli_stops_when_the_weight_is_missing_and_writes_nothing(tmp_path: Path):
    image = tmp_path / "portrait.png"
    _write_image(image)
    out_labels = tmp_path / "labels.png"
    with pytest.raises(SystemExit, match="model not found"):
        run_human_parsing.main(
            [
                "--image",
                str(image),
                "--model",
                str(tmp_path / "absent.onnx"),
                "--out-labels",
                str(out_labels),
                "--stats-json",
                str(tmp_path / "stats.json"),
            ]
        )
    assert not out_labels.exists()
    assert not (tmp_path / "stats.json").exists()


def test_cli_refuses_an_unverified_weight_and_writes_nothing(tmp_path: Path):
    image = tmp_path / "portrait.png"
    _write_image(image)
    model = tmp_path / "schp-atr-18.onnx"
    model.write_bytes(b"unverified")
    out_labels = tmp_path / "labels.png"
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        run_human_parsing.main(
            [
                "--image",
                str(image),
                "--model",
                str(model),
                "--out-labels",
                str(out_labels),
                "--stats-json",
                str(tmp_path / "stats.json"),
            ]
        )
    assert not out_labels.exists()
    assert not (tmp_path / "stats.json").exists()
