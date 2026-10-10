"""Unit tests for the local ``auto`` domain heuristic."""

from pathlib import Path

from app.domains.detector import detect_domain_type, skin_tone_ratio
from tests.conftest import make_image_bytes, make_landscape_bytes, make_portrait_bytes


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_portrait_frame_is_detected_as_portrait(tmp_path: Path) -> None:
    path = _write(tmp_path, "portrait.png", make_portrait_bytes())

    assert detect_domain_type(path) == "portrait"


def test_landscape_frame_is_detected_as_general(tmp_path: Path) -> None:
    path = _write(tmp_path, "landscape.png", make_landscape_bytes())

    assert detect_domain_type(path) == "general"


def test_tall_landscape_frame_is_not_a_portrait(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "tall-scenery.png",
        make_landscape_bytes(size=(640, 960)),
    )

    assert detect_domain_type(path) == "general"


def test_flat_colour_frame_is_not_a_portrait(tmp_path: Path) -> None:
    path = _write(tmp_path, "flat.png", make_image_bytes(size=(640, 960)))

    assert detect_domain_type(path) == "general"


def test_portrait_frame_has_more_skin_tone_than_landscape(
    tmp_path: Path,
) -> None:
    from PIL import Image

    with Image.open(_write(tmp_path, "portrait.png", make_portrait_bytes())) as image:
        portrait_ratio = skin_tone_ratio(image.convert("RGB"))
    with Image.open(_write(tmp_path, "landscape.png", make_landscape_bytes())) as image:
        landscape_ratio = skin_tone_ratio(image.convert("RGB"))

    assert portrait_ratio > 0.12
    assert landscape_ratio == 0.0
