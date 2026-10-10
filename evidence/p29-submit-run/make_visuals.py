"""Triptychs for the single authorized paid render: original / last failure / this run."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-submit-run"
PANELS = [
    ("ORIGINAL", ROOT / "data/inputs/IMG_3651.jpg"),
    ("LAST FAILED (pale ghost)", ROOT / "artifacts/p29-fix-run/result_0.png"),
    ("THIS RUN (task 2108330225116082177)", OUT / "final.png"),
]
BG = (24, 24, 28)

CROPS = {
    "triptych": None,
    "neck_zoom3x": (280, 430, 640, 700),
    "hairline_zoom3x": (200, 60, 720, 420),
    "collar_zoom3x": (250, 560, 620, 900),
}


def build(name: str, crop: tuple[int, int, int, int] | None, scale: int) -> None:
    images = []
    for label, path in PANELS:
        im = Image.open(path).convert("RGB")
        if crop:
            im = im.crop(crop)
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        images.append((label, im))
    pad, header = 8, 22
    width = sum(im.width for _, im in images) + pad * (len(images) + 1)
    height = max(im.height for _, im in images) + header + pad * 2
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, im in images:
        canvas.paste(im, (x, header + pad))
        draw.text((x + 2, 5), label, fill=(235, 235, 235))
        x += im.width + pad
    target = OUT / f"{name}.png"
    canvas.save(target)
    print(f"{target} {canvas.size}")


if __name__ == "__main__":
    for name, crop in CROPS.items():
        build(name, crop, 3 if crop else 1)
