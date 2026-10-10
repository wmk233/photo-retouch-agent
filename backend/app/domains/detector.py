"""Best-effort local domain inference for the ``auto`` domain type.

This is intentionally a small heuristic rather than face recognition: it looks
at the framing (tall/square vs wide) and at how much skin-tone area sits in the
middle of the frame, where a portrait subject normally is. A caller that wants
a guaranteed domain keeps passing an explicit ``domainType``; a caller that has
no idea (the web UI) passes ``auto`` and gets a classification instead of a
hardcoded assumption.
"""

from pathlib import Path

from PIL import Image

AUTO_DOMAIN_TYPE = "auto"
PORTRAIT_DOMAIN_TYPE = "portrait"
GENERAL_DOMAIN_TYPE = "general"

_CENTRE_CROP_RATIO = 0.7
_SAMPLE_SIZE = 64
# A face/upper-body crop normally carries at least this much skin-tone area.
_SKIN_RATIO_THRESHOLD = 0.12
# Vertical framing: tall or roughly square, but not a wide landscape.
_MIN_VERTICAL_RATIO = 0.92


def _is_skin_tone(red: int, green: int, blue: int) -> bool:
    return (
        red > 95
        and green > 40
        and blue > 20
        and max(red, green, blue) - min(red, green, blue) > 15
        and abs(red - green) > 15
        and red > green
        and red > blue
    )


def skin_tone_ratio(image: Image.Image) -> float:
    """Share of skin-tone pixels in the centre of the frame, in ``0.0..1.0``."""
    width, height = image.size
    margin_x = width * (1 - _CENTRE_CROP_RATIO) / 2
    margin_y = height * (1 - _CENTRE_CROP_RATIO) / 2
    centre = image.crop(
        (
            int(margin_x),
            int(margin_y),
            int(width - margin_x),
            int(height - margin_y),
        )
    )
    sample = centre.resize((_SAMPLE_SIZE, _SAMPLE_SIZE))
    pixels = sample.tobytes()
    skin_pixels = sum(
        1
        for index in range(0, len(pixels), 3)
        if _is_skin_tone(pixels[index], pixels[index + 1], pixels[index + 2])
    )
    return skin_pixels / (_SAMPLE_SIZE * _SAMPLE_SIZE)


def detect_domain_type(image_path: Path) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        ratio = skin_tone_ratio(rgb)

    is_vertical = height >= width * _MIN_VERTICAL_RATIO
    if is_vertical and ratio >= _SKIN_RATIO_THRESHOLD:
        return PORTRAIT_DOMAIN_TYPE
    return GENERAL_DOMAIN_TYPE
