"""Observable portrait facts extracted from the uploaded photo.

The AI stylist must reason about what is actually visible in the picture and it
must never invent personal attributes. This module therefore only measures things
a viewer could point at: facial proportions, how much hair covers the head, if
glasses seem to be present, and the dominant colours of the clothing.

Every measurement is a coarse heuristic computed with Pillow on a downscaled copy
of the image. The labels are deliberately hedged ("偏", "约", "未识别") so the
downstream prompts never claim more certainty than the pixels support.

Nothing here infers gender, age, ethnicity, occupation, personality or any other
private attribute.
"""

from __future__ import annotations

from pathlib import Path
from statistics import pstdev
from typing import Any

from PIL import Image

from app.domains.detector import _is_skin_tone

_SAMPLE_MAX = 128

# Vertical bands as a share of the image, using a normal head-and-shoulders crop.
_FACE_X = (0.18, 0.82)
_FACE_Y = (0.06, 0.68)
_FACE_ROW_SKIN_MIN = 0.22
_HAIR_X = (0.16, 0.84)
_HAIR_Y = (0.02, 0.55)
_HAIR_ROW_MIN = 0.30
_BODY_X = (0.08, 0.92)
_BODY_Y_MAX = 0.98
_BODY_TOP_OFFSET = 0.04
_BACKGROUND_DISTANCE = 60.0

# A small, named palette so the facts stay readable in the UI and the prompts.
_NAMED_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("黑色", (26, 26, 30)),
    ("深灰", (78, 80, 84)),
    ("灰色", (140, 142, 146)),
    ("白色", (238, 238, 236)),
    ("米白", (226, 214, 196)),
    ("卡其", (176, 152, 116)),
    ("浅蓝", (150, 178, 208)),
    ("藏蓝", (44, 60, 96)),
    ("牛仔蓝", (86, 116, 158)),
    ("红色", (176, 52, 52)),
    ("酒红", (116, 40, 54)),
    ("粉色", (222, 158, 176)),
    ("橙色", (214, 126, 62)),
    ("黄色", (218, 190, 84)),
    ("绿色", (74, 128, 86)),
    ("橄榄绿", (110, 118, 70)),
    ("紫色", (128, 96, 158)),
    ("棕色", (122, 88, 62)),
)

_COLOR_FAMILY = {
    "黑色": "中性",
    "深灰": "中性",
    "灰色": "中性",
    "白色": "中性",
    "米白": "中性",
    "卡其": "暖色",
    "红色": "暖色",
    "酒红": "暖色",
    "粉色": "暖色",
    "橙色": "暖色",
    "黄色": "暖色",
    "棕色": "暖色",
    "浅蓝": "冷色",
    "藏蓝": "冷色",
    "牛仔蓝": "冷色",
    "绿色": "冷色",
    "橄榄绿": "冷色",
    "紫色": "冷色",
}


def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return float(sum(abs(int(a) - int(b)) for a, b in zip(left, right)))


def _luminance(pixel: tuple[int, int, int]) -> float:
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def _nearest_color_name(pixel: tuple[int, int, int]) -> str:
    return min(_NAMED_COLORS, key=lambda item: _distance(pixel, item[1]))[0]


def _value_label(pixel: tuple[int, int, int]) -> str:
    light = _luminance(pixel)
    if light >= 190:
        return "浅色"
    if light >= 110:
        return "中间色"
    return "深色"


def _load_sample(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = _SAMPLE_MAX / max(width, height, 1)
        if scale < 1:
            rgb = rgb.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.BILINEAR,
            )
        return rgb


def _background_color(pixels, width: int, height: int) -> tuple[int, int, int]:
    patch = max(2, int(min(width, height) * 0.05))
    samples: list[tuple[int, int, int]] = []
    for origin_x, origin_y in (
        (0, 0),
        (width - patch, 0),
        (0, height - patch),
        (width - patch, height - patch),
    ):
        for x in range(origin_x, min(width, origin_x + patch)):
            for y in range(origin_y, min(height, origin_y + patch)):
                samples.append(pixels[x, y])
    if not samples:
        return (128, 128, 128)
    count = len(samples)
    return (
        sum(pixel[0] for pixel in samples) // count,
        sum(pixel[1] for pixel in samples) // count,
        sum(pixel[2] for pixel in samples) // count,
    )


def _face_shape(ratio: float, taper: float) -> tuple[str, str]:
    if ratio <= 1.32 and taper >= 0.82:
        return "round", "面部宽高比例接近，下颌线条圆润"
    if ratio >= 1.72:
        return "oblong", "面部纵向比例明显大于横向"
    if taper <= 0.70:
        return "heart", "颧骨区域最宽，向下颌明显收窄"
    if taper >= 0.90 and ratio <= 1.60:
        return "square", "颧骨与下颌宽度接近，轮廓偏方正"
    return "oval", "面部纵横比适中，轮廓过渡自然"


def _shape_adjust(shape: str) -> str:
    return {
        "round": "纵向线条和顶部体积能拉长面部比例",
        "square": "柔和的层次和自然卷度能软化下颌线条",
        "oblong": "两侧横向体积可以平衡面部的纵向长度",
        "heart": "保留下颌区域的利落感能突出颧骨优势",
        "oval": "可以保留原本均衡的面部比例",
    }.get(shape, "可以保留原本均衡的面部比例")


def _styling_directions(facts: dict[str, Any]) -> list[str]:
    directions: list[str] = [_shape_adjust(str(facts.get("face_shape", "oval")))]
    volume = str(facts.get("hair_volume_label", ""))
    if "偏少" in volume:
        directions.append("增加顶部蓬松度，让轮廓更饱满")
    elif "偏多" in volume:
        directions.append("用高层次结构减轻发量带来的厚重感")
    if facts.get("wears_glasses"):
        directions.append("眼镜是辨识元素，新造型需保留或换成风格一致的镜框")
    hair_length = str(facts.get("current_hair_length", "unknown"))
    if hair_length == "long":
        directions.append("长发可尝试整体提拉的高马尾或利落短发形成对比")
    elif hair_length == "short":
        directions.append("短发已成立，可在发色和纹理上做更大胆的尝试")
    tone = str(facts.get("skin_temperature_label", ""))
    if tone and tone != "未识别":
        directions.append(f"肤色呈{tone}，发色与服装配色可沿用同向冷暖关系")
    return directions[:5]


def _neutral_facts() -> dict[str, Any]:
    return {
        "face_shape": "oval",
        "face_shape_reason": "照片可用信息不足，按通用比例处理",
        "forehead_height_percent": 0.0,
        "forehead_height_label": "未能从照片确认",
        "face_width_ratio": 0.0,
        "face_length_label": "未识别",
        "cheek_fullness": "未识别",
        "hair_volume_label": "未识别",
        "hair_coverage_label": "未识别",
        "current_hair_length": "unknown",
        "current_hair_length_label": "未识别",
        "hair_color_name": "未识别",
        "hair_texture_label": "未识别",
        "wears_glasses": False,
        "glasses_confidence": 0.0,
        "eyewear_label": "未识别",
        "skin_tone_rgb": [0, 0, 0],
        "skin_tone_label": "未识别",
        "skin_temperature_label": "未识别",
        "upper_body_color_name": "未识别",
        "upper_body_color_rgb": [0, 0, 0],
        "upper_body_color_family": "未识别",
        "upper_body_value": "未识别",
        "upper_body_palette": [],
        "upper_body_frame": "未识别",
        "frame_type": "未识别",
        "head_shoulder_label": "未识别",
        "styling_directions": [],
    }


def derive_face_facts(image_path: Path) -> dict[str, Any]:
    """Measure the visible features a stylist would describe out loud."""
    sample = _load_sample(image_path)
    width, height = sample.size
    if width < 8 or height < 8:
        return _neutral_facts()

    pixels = sample.load()
    background = _background_color(pixels, width, height)

    face_left = int(_FACE_X[0] * width)
    face_right = max(face_left + 1, int(_FACE_X[1] * width))
    face_top_limit = int(_FACE_Y[0] * height)
    face_bottom_limit = max(face_top_limit + 1, int(_FACE_Y[1] * height))
    band_width = face_right - face_left

    skin_rows = [0] * height
    skin_extent: list[tuple[int, int] | None] = [None] * height
    skin_total = [0, 0, 0]
    skin_count = 0
    for y in range(face_top_limit, face_bottom_limit):
        row_min: int | None = None
        row_max: int | None = None
        for x in range(face_left, face_right):
            pixel = pixels[x, y]
            if not _is_skin_tone(pixel[0], pixel[1], pixel[2]):
                continue
            skin_rows[y] += 1
            skin_count += 1
            skin_total[0] += pixel[0]
            skin_total[1] += pixel[1]
            skin_total[2] += pixel[2]
            row_min = x if row_min is None else min(row_min, x)
            row_max = x if row_max is None else max(row_max, x)
        if row_min is not None and row_max is not None:
            skin_extent[y] = (row_min, row_max)

    skin_row_min = _FACE_ROW_SKIN_MIN * band_width
    face_rows = [
        y
        for y in range(face_top_limit, face_bottom_limit)
        if skin_rows[y] >= skin_row_min
    ]
    if face_rows:
        face_top, face_bottom = face_rows[0], face_rows[-1]
    else:
        face_top, face_bottom = face_top_limit, face_bottom_limit
    face_height = max(1, face_bottom - face_top + 1)

    widest_row = face_top
    face_width_px = 1
    for y in range(face_top, face_bottom + 1):
        extent = skin_extent[y]
        if extent is None:
            continue
        row_width = extent[1] - extent[0]
        if row_width > face_width_px:
            face_width_px = row_width
            widest_row = y

    upper_edge = face_top + int(face_height * 0.45)
    lower_edge = face_bottom - int(face_height * 0.28)
    upper_width = 1
    lower_width = 1
    for y in range(face_top, max(face_top + 1, upper_edge)):
        extent = skin_extent[y]
        if extent is not None:
            upper_width = max(upper_width, extent[1] - extent[0])
    for y in range(min(lower_edge, face_bottom), face_bottom + 1):
        extent = skin_extent[y]
        if extent is not None:
            lower_width = max(lower_width, extent[1] - extent[0])

    ratio = face_height / face_width_px
    taper = lower_width / max(upper_width, 1)
    face_shape, face_shape_reason = _face_shape(ratio, taper)

    if ratio >= 1.70:
        face_length_label = "偏修长"
    elif ratio <= 1.34:
        face_length_label = "偏紧凑"
    else:
        face_length_label = "匀称"

    widest_position = (widest_row - face_top) / face_height
    if widest_position < 0.38:
        cheek_fullness = "上庭（颧骨以上）最宽"
    elif widest_position <= 0.62:
        cheek_fullness = "中庭（颧骨至面颊）最宽"
    else:
        cheek_fullness = "下庭（下颌区域）最宽"

    # --- hair ---------------------------------------------------------------
    hair_left = int(_HAIR_X[0] * width)
    hair_right = max(hair_left + 1, int(_HAIR_X[1] * width))
    hair_top = int(_HAIR_Y[0] * height)
    hair_bottom_limit = min(
        int(_HAIR_Y[1] * height),
        face_bottom + max(1, int(0.06 * height)),
    )
    hair_bottom_limit = max(hair_top + 1, hair_bottom_limit)
    hair_band_width = max(1, hair_right - hair_left)
    hair_band_height = max(1, hair_bottom_limit - hair_top)

    hair_rows = [0] * height
    hair_pixels: list[tuple[int, int, int]] = []
    for y in range(hair_top, hair_bottom_limit):
        for x in range(hair_left, hair_right):
            pixel = pixels[x, y]
            if _is_skin_tone(pixel[0], pixel[1], pixel[2]):
                continue
            if _distance(pixel, background) <= _BACKGROUND_DISTANCE:
                continue
            hair_rows[y] += 1
            hair_pixels.append(pixel)

    coverage = len(hair_pixels) / (hair_band_width * hair_band_height)
    if coverage >= 0.40:
        hair_coverage_label, hair_volume_label = "头发覆盖面积偏多", "发量可见偏多"
    elif coverage >= 0.16:
        hair_coverage_label, hair_volume_label = "头发覆盖面积中等", "发量可见中等"
    elif coverage >= 0.04:
        hair_coverage_label, hair_volume_label = "头发覆盖面积偏少", "发量可见偏少"
    else:
        hair_coverage_label, hair_volume_label = "未识别到明显头发", "未识别"

    hair_row_min = _HAIR_ROW_MIN * hair_band_width
    hair_rows_with_hair = [
        y for y in range(hair_top, hair_bottom_limit) if hair_rows[y] >= hair_row_min
    ]
    hair_bottom_row = hair_rows_with_hair[-1] if hair_rows_with_hair else None

    if hair_bottom_row is None:
        forehead_height_percent = 0.0
        forehead_height_label = "未能从照片确认发际线"
    else:
        forehead_ratio = (face_top - hair_bottom_row) / face_height
        forehead_height_percent = round(max(0.0, min(0.6, forehead_ratio)) * 100, 1)
        if forehead_ratio >= 0.30:
            forehead_height_label = "额前较开阔，发际线偏高"
        elif forehead_ratio >= 0.18:
            forehead_height_label = "额头高度中等"
        else:
            forehead_height_label = "头发覆盖额前，发际线偏低"

    if hair_pixels:
        hair_mean = (
            sum(pixel[0] for pixel in hair_pixels) // len(hair_pixels),
            sum(pixel[1] for pixel in hair_pixels) // len(hair_pixels),
            sum(pixel[2] for pixel in hair_pixels) // len(hair_pixels),
        )
        hair_color_name = _nearest_color_name(hair_mean)
        hair_deviation = pstdev([_luminance(pixel) for pixel in hair_pixels])
        hair_texture_label = (
            "可见发丝层次与明暗变化"
            if hair_deviation >= 26
            else "发色较均匀、整体顺贴"
        )
    else:
        hair_color_name = "未识别"
        hair_texture_label = "未识别"

    side_left = (int(0.04 * width), max(int(0.04 * width) + 1, int(0.20 * width)))
    side_right = (int(0.80 * width), max(int(0.80 * width) + 1, int(0.96 * width)))
    side_bottom_limit = min(
        height - 1, face_bottom + max(1, int(0.45 * face_height))
    )
    side_width = max(1, side_left[1] - side_left[0])
    side_hair_bottom: int | None = None
    for y in range(face_top, side_bottom_limit):
        count = 0
        for left, right in (side_left, side_right):
            for x in range(left, right):
                pixel = pixels[x, y]
                if _is_skin_tone(pixel[0], pixel[1], pixel[2]):
                    continue
                if _distance(pixel, background) > _BACKGROUND_DISTANCE:
                    count += 1
        if count >= 0.10 * side_width * 2:
            side_hair_bottom = y

    if side_hair_bottom is None:
        current_hair_length = "unknown"
        current_hair_length_label = "长度未识别"
    else:
        extension = (side_hair_bottom - face_bottom) / face_height
        if extension >= 0.45:
            current_hair_length = "long"
            current_hair_length_label = "长发（明显过肩）"
        elif extension >= 0.12:
            current_hair_length = "medium"
            current_hair_length_label = "中长发（及肩至锁骨）"
        else:
            current_hair_length = "short"
            current_hair_length_label = "短发（耳上至下颌）"

    # --- glasses ------------------------------------------------------------
    eye_top = face_top + int(face_height * 0.32)
    eye_bottom = face_top + int(face_height * 0.52)
    dark_pixels = 0
    eye_pixels = 0
    for y in range(eye_top, max(eye_top + 1, eye_bottom)):
        for x in range(face_left, face_right):
            pixel = pixels[x, y]
            eye_pixels += 1
            if max(pixel) < 95 and (max(pixel) - min(pixel)) < 48:
                dark_pixels += 1
    dark_ratio = dark_pixels / max(eye_pixels, 1)
    glasses_confidence = round(min(1.0, dark_ratio / 0.14), 2)
    wears_glasses = glasses_confidence >= 0.55
    eyewear_label = (
        "疑似佩戴眼镜（请在生成时保留镜框特征）" if wears_glasses else "未见明显眼镜轮廓"
    )

    # --- skin ---------------------------------------------------------------
    if skin_count:
        skin_mean = (
            skin_total[0] // skin_count,
            skin_total[1] // skin_count,
            skin_total[2] // skin_count,
        )
    else:
        skin_mean = (0, 0, 0)
    skin_light = _luminance(skin_mean)
    if not skin_count:
        skin_tone_label, skin_temperature_label = "未识别", "未识别"
    else:
        if skin_light >= 198:
            skin_tone_label = "偏白皙"
        elif skin_light >= 162:
            skin_tone_label = "自然"
        elif skin_light >= 130:
            skin_tone_label = "健康小麦"
        else:
            skin_tone_label = "偏深"
        warmth = skin_mean[0] - skin_mean[2]
        skin_temperature_label = "暖调" if warmth > 28 else "冷调" if warmth < 10 else "中性"

    # --- clothing -----------------------------------------------------------
    body_top = min(height - 1, face_bottom + max(1, int(_BODY_TOP_OFFSET * height)))
    body_bottom = max(body_top + 1, int(_BODY_Y_MAX * height))
    body_left = int(_BODY_X[0] * width)
    body_right = max(body_left + 1, int(_BODY_X[1] * width))
    histogram: dict[str, int] = {}
    body_colors: list[tuple[int, int, int]] = []
    first_half_extent = 0
    second_half_extent = 0
    split = body_top + (body_bottom - body_top) // 2
    for y in range(body_top, body_bottom):
        row_min: int | None = None
        row_max: int | None = None
        for x in range(body_left, body_right):
            pixel = pixels[x, y]
            if _is_skin_tone(pixel[0], pixel[1], pixel[2]):
                continue
            if _distance(pixel, background) <= _BACKGROUND_DISTANCE:
                continue
            name = _nearest_color_name(pixel)
            histogram[name] = histogram.get(name, 0) + 1
            body_colors.append(pixel)
            row_min = x if row_min is None else min(row_min, x)
            row_max = x if row_max is None else max(row_max, x)
        if row_min is not None and row_max is not None:
            extent = row_max - row_min
            if y < split:
                first_half_extent = max(first_half_extent, extent)
            else:
                second_half_extent = max(second_half_extent, extent)

    if histogram:
        ranked = sorted(histogram.items(), key=lambda item: item[1], reverse=True)
        palette_names = [name for name, _ in ranked[:2]]
        primary_name = ranked[0][0]
        primary_pixels = [p for p in body_colors if _nearest_color_name(p) == primary_name]
        upper_body_color_rgb = [
            sum(pixel[channel] for pixel in primary_pixels) // max(1, len(primary_pixels))
            for channel in range(3)
        ]
        upper_body_color_family = _COLOR_FAMILY.get(primary_name, "中性")
        upper_body_value = _value_label(tuple(upper_body_color_rgb))
        if second_half_extent > first_half_extent * 1.18:
            upper_body_frame = "下摆更宽的宽松版型"
        elif second_half_extent < first_half_extent * 0.85:
            upper_body_frame = "上宽下收的收腰版型"
        else:
            upper_body_frame = "轮廓顺直的合身版型"
    else:
        palette_names = []
        primary_name = "未识别"
        upper_body_color_rgb = [0, 0, 0]
        upper_body_color_family = "未识别"
        upper_body_value = "未识别"
        upper_body_frame = "未识别"

    face_ratio = face_height / height
    if face_ratio >= 0.42:
        frame_type = "近景头像"
    elif face_ratio >= 0.24:
        frame_type = "半身"
    else:
        frame_type = "大半身或全身"

    shoulder_extent = max(first_half_extent, second_half_extent)
    shoulder_ratio = shoulder_extent / face_width_px
    if not histogram:
        head_shoulder_label = "未识别"
    elif shoulder_ratio >= 2.6:
        head_shoulder_label = "肩线舒展，画面有上半身信息"
    elif shoulder_ratio >= 1.8:
        head_shoulder_label = "肩线适中"
    else:
        head_shoulder_label = "近景构图，肩线未完整入镜"

    facts: dict[str, Any] = {
        "face_shape": face_shape,
        "face_shape_reason": face_shape_reason,
        "forehead_height_percent": forehead_height_percent,
        "forehead_height_label": forehead_height_label,
        "face_width_ratio": round(face_width_px / width, 3),
        "face_length_label": face_length_label,
        "cheek_fullness": cheek_fullness,
        "hair_volume_label": hair_volume_label,
        "hair_coverage_label": hair_coverage_label,
        "current_hair_length": current_hair_length,
        "current_hair_length_label": current_hair_length_label,
        "hair_color_name": hair_color_name,
        "hair_texture_label": hair_texture_label,
        "wears_glasses": wears_glasses,
        "glasses_confidence": glasses_confidence,
        "eyewear_label": eyewear_label,
        "skin_tone_rgb": list(skin_mean),
        "skin_tone_label": skin_tone_label,
        "skin_temperature_label": skin_temperature_label,
        "upper_body_color_name": primary_name,
        "upper_body_color_rgb": list(upper_body_color_rgb),
        "upper_body_color_family": upper_body_color_family,
        "upper_body_value": upper_body_value,
        "upper_body_palette": palette_names,
        "upper_body_frame": upper_body_frame,
        "frame_type": frame_type,
        "head_shoulder_label": head_shoulder_label,
    }
    if not skin_count:
        # Without a detected skin-tone region there is no face to describe, so
        # every face/hair reading is noise. Keep the measurable clothing colours
        # and report the rest as unknown rather than inventing a hairstyle.
        facts.update(
            {
                "face_shape": "oval",
                "face_shape_reason": "照片中未识别到清晰的正面肤色区域，按通用比例处理",
                "forehead_height_percent": 0.0,
                "forehead_height_label": "未能从照片确认发际线",
                "face_width_ratio": 0.0,
                "face_length_label": "未识别",
                "cheek_fullness": "未识别",
                "hair_volume_label": "未识别",
                "hair_coverage_label": "未识别",
                "current_hair_length": "unknown",
                "current_hair_length_label": "长度未识别",
                "hair_color_name": "未识别",
                "hair_texture_label": "未识别",
                "frame_type": "未识别",
                "head_shoulder_label": "未识别",
            }
        )
    facts["styling_directions"] = _styling_directions(facts)
    return facts
