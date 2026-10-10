"""The fal.ai inpaint preflight: it must refuse a request it cannot stand behind.

The preflight is the last free step before a billed queue POST, so its checks
are the only thing between a wrong payload and a wasted generation. These tests
drive the pure helpers with deliberately broken requests and assert that each
one is reported rather than waved through.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import preflight_fal_inpaint as tool


def _write_image(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array).save(path)
    return path


def _data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@pytest.fixture()
def photo_and_mask(tmp_path: Path) -> tuple[Path, Path]:
    photo = np.full((40, 30, 3), 120, dtype=np.uint8)
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[5:20, 5:25] = 255
    return (
        _write_image(tmp_path / "photo.png", photo),
        _write_image(tmp_path / "mask.png", mask),
    )


def _payload(photo: Path, mask: Path) -> dict:
    return {
        "prompt": "restyle the hair",
        "image_url": _data_uri(photo),
        "mask_url": _data_uri(mask),
        "output_format": "png",
        "num_images": 1,
        "sync_mode": False,
    }


# ------------------------------------------------------------- data URI guard


def test_decode_data_uri_round_trips_the_bytes(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "x.png", np.zeros((4, 4), dtype=np.uint8))
    mime, data = tool._decode_data_uri(_data_uri(path))
    assert mime == "image/png"
    assert data == path.read_bytes()


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/photo.png",
        "data:image/png,plaintext-not-base64",
        "data:image/png;base64,@@not-base64@@",
        None,
    ],
)
def test_decode_data_uri_rejects_anything_that_is_not_an_inline_image(value: object) -> None:
    with pytest.raises(ValueError):
        tool._decode_data_uri(value)


# ------------------------------------------------------------ contract checks


def test_a_well_formed_request_reports_no_problems(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    report = tool._contract_report(_payload(photo, mask), photo, mask, (30, 40))
    assert report["problems"] == []


def test_the_two_inputs_must_not_be_the_same_picture(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    payload = _payload(photo, mask)
    payload["mask_url"] = payload["image_url"]
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("same payload" in problem for problem in report["problems"])


def test_a_missing_mask_is_reported(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    payload = _payload(photo, mask)
    payload.pop("mask_url")
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("mask_url" in problem for problem in report["problems"])


def test_num_images_above_one_is_reported(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    payload = _payload(photo, mask)
    payload["num_images"] = 2
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("num_images" in problem for problem in report["problems"])


def test_a_mask_in_other_coordinates_is_reported(tmp_path: Path) -> None:
    photo = _write_image(tmp_path / "photo.png", np.full((40, 30, 3), 10, dtype=np.uint8))
    # The mask is saved at half the photo's size: it is not in photo coordinates.
    mask_array = np.zeros((20, 15), dtype=np.uint8)
    mask_array[2:10, 2:10] = 255
    mask = _write_image(tmp_path / "mask.png", mask_array)
    payload = _payload(photo, mask)
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("own coordinates" in problem for problem in report["problems"])


def test_a_grey_mask_is_reported(tmp_path: Path) -> None:
    photo = _write_image(tmp_path / "photo.png", np.full((40, 30, 3), 10, dtype=np.uint8))
    mask_array = np.zeros((40, 30), dtype=np.uint8)
    mask_array[5:20, 5:25] = 200  # neither keep nor repaint
    mask = _write_image(tmp_path / "mask.png", mask_array)
    payload = _payload(photo, mask)
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("strictly 0/255" in problem for problem in report["problems"])


def test_a_credential_in_the_payload_is_reported(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    payload = _payload(photo, mask)
    payload["api_key"] = "should-never-travel-here"
    report = tool._contract_report(payload, photo, mask, (30, 40))
    assert any("credential" in problem for problem in report["problems"])


def test_an_omitted_image_size_is_noted_not_failed(photo_and_mask: tuple[Path, Path]) -> None:
    photo, mask = photo_and_mask
    report = tool._contract_report(_payload(photo, mask), photo, mask, (30, 40))
    assert any("image_size" in note for note in report["notes"])
    assert not any("image_size" in problem for problem in report["problems"])


# ------------------------------------------------------- cached label segmenter


def test_cached_segmenter_serves_the_label_map_it_was_given() -> None:
    labels = np.zeros((40, 30), dtype=np.uint8)
    segmenter = tool._CachedLabelMapSegmenter(labels)
    assert np.array_equal(segmenter.segment(np.zeros((40, 30, 3), dtype=np.uint8)), labels)
    assert segmenter.close() is None


def test_cached_segmenter_rejects_a_label_map_from_another_frame() -> None:
    segmenter = tool._CachedLabelMapSegmenter(np.zeros((40, 30), dtype=np.uint8))
    with pytest.raises(ValueError):
        segmenter.segment(np.zeros((80, 60, 3), dtype=np.uint8))


def test_pixels_counts_only_set_pixels() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :5] = 255
    assert tool._pixels(mask) == 5
