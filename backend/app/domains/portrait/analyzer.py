from pathlib import Path

from PIL import Image, ImageStat

from app.schemas.analysis import PhotoAnalysis, SubjectSummary


class PortraitAnalyzer:
    domain_type = "portrait"

    def analyze(self, image_id: str, image_path: Path) -> PhotoAnalysis:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            stat = ImageStat.Stat(rgb.resize((72, 72)))

        channels = stat.mean
        brightness = sum(channels) / 3
        warmth = channels[0] - channels[2]
        contrast = sum(stat.stddev) / 3

        is_portrait = height >= width * 1.08
        is_square = abs(width - height) < max(width, height) * 0.08
        is_dark = brightness < 98
        is_bright = brightness > 185
        is_warm = warmth > 14
        is_cool = warmth < -10
        low_contrast = contrast < 42
        low_resolution = width < 900 or height < 900

        lighting_issues = [
            is_dark and "整体偏暗",
            is_bright and "高光偏亮",
            is_cool and "色温偏冷",
            is_warm and "色温偏暖",
            low_contrast and "层次略平",
        ]
        lighting = [item for item in lighting_issues if item] or ["光线基础良好"]

        background_issues = [
            not is_portrait and "背景信息较多",
            low_contrast and "主体和背景分离度一般",
            is_portrait and "边缘干扰可进一步弱化",
        ]
        background = [item for item in background_issues if item] or ["背景干扰较少"]

        scene_type = "人像 / 自拍" if is_portrait else "头像 / 社媒图" if is_square else "旅行 / 街拍"

        return PhotoAnalysis(
            image_id=image_id,
            domain_type=self.domain_type,
            scene_type=scene_type,
            subjects=SubjectSummary(
                count=1,
                position="中心区域" if is_portrait or is_square else "中间偏左/偏右需模型确认",
                face_visibility="较高" if is_portrait else "中等",
            ),
            lighting_issues=lighting,
            background_issues=background,
            portrait_suggestions=[
                "提升面部亮度" if is_dark else "保持自然肤色",
                "轻度统一肤色",
                "保留皮肤纹理",
                "增强眼神光和面部层次" if low_contrast else "轻微增强五官清晰度",
            ],
            composition_suggestions=[
                "适合保留竖版构图" if is_portrait else "可裁成头像或社媒封面",
                "主体位置保持稳定",
                "边缘杂物适度弱化",
            ],
            recommended_styles=[
                "自然",
                "精致头像" if is_portrait else "旅行氛围",
                "电影感" if is_warm else "清透社媒",
            ],
            risk_flags=[
                item
                for item in [
                    low_resolution and "原图分辨率偏低",
                    is_bright and "过曝区域需保守处理",
                    is_dark and "暗部提亮可能带来噪点",
                ]
                if item
            ],
        )
