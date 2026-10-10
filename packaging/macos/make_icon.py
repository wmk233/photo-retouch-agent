#!/usr/bin/env python3
"""生成 AppIcon.icns（无需外部素材）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CANVAS = 1024
INSET = 100
RADIUS = 225
# 图标主字是汉字，优先取中黑体；后面的拉丁字体只作兜底。
FONT_CANDIDATES = (
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 1),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path, index in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:
            continue
    return ImageFont.load_default()


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def build_master() -> Image.Image:
    box = CANVAS - INSET * 2
    gradient = Image.new("RGB", (box, box))
    draw = ImageDraw.Draw(gradient)
    top = (189, 152, 255)
    bottom = (70, 44, 194)
    for y in range(box):
        ratio = y / (box - 1)
        eased = ratio ** 0.85
        draw.line(
            [(0, y), (box, y)],
            fill=tuple(round(top[i] + (bottom[i] - top[i]) * eased) for i in range(3)),
        )

    glow = Image.new("L", (box, box), 0)
    ImageDraw.Draw(glow).ellipse(
        (-box * 0.25, -box * 0.55, box * 1.05, box * 0.45), fill=90
    )
    glow = glow.filter(ImageFilter.GaussianBlur(box * 0.09))
    gradient = Image.composite(Image.new("RGB", (box, box), (255, 255, 255)), gradient, glow.point(lambda v: int(v * 0.45)))

    tile = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    tile.paste(gradient, (0, 0), rounded_mask(box, RADIUS))

    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (INSET, INSET + 12, CANVAS - INSET, CANVAS - INSET + 12), radius=RADIUS, fill=(46, 26, 122, 80)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(26))

    master = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    master.alpha_composite(shadow)
    master.alpha_composite(tile, (INSET, INSET))

    label = "图"
    font = load_font(420)
    text_layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    ImageDraw.Draw(text_layer).text(
        (CANVAS / 2, CANVAS / 2 - 6),
        label,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="mm",
    )
    master.alpha_composite(text_layer.filter(ImageFilter.GaussianBlur(0.6)))
    return master


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    master = build_master()
    iconset = out_dir / "AppIcon.iconset"
    if iconset.exists():
        subprocess.run(["rm", "-rf", str(iconset)], check=True)
    iconset.mkdir(parents=True)

    for size in (16, 32, 64, 128, 256, 512):
        master.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        master.resize((size * 2, size * 2), Image.LANCZOS).save(
            iconset / f"icon_{size}x{size}@2x.png"
        )
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_dir / "AppIcon.icns")], check=True)
    master.save(out_dir / "AppIcon-1024.png")
    print(out_dir / "AppIcon.icns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
