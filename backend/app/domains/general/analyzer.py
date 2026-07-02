from pathlib import Path

from PIL import Image, ImageStat

from app.schemas.analysis import PhotoAnalysis, SubjectSummary


class GeneralAnalyzer:
    domain_type = "general"

    def analyze(self, image_id: str, image_path: Path) -> PhotoAnalysis:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            stat = ImageStat.Stat(rgb.resize((72, 72)))

        channels = stat.mean
        brightness = sum(channels) / 3
        warmth = channels[0] - channels[2]
        contrast = sum(stat.stddev) / 3

        is_dark = brightness < 98
        is_bright = brightness > 185
        is_warm = warmth > 14
        is_cool = warmth < -10
        low_contrast = contrast < 42
        low_resolution = width < 900 or height < 900
        orientation = (
            "横向构图"
            if width > height * 1.12
            else "竖向构图"
            if height > width * 1.12
            else "方形构图"
        )

        lighting = [
            item
            for item in [
                is_dark and "整体偏暗",
                is_bright and "高光偏亮",
                is_cool and "色温偏冷",
                is_warm and "色温偏暖",
                low_contrast and "画面层次略平",
            ]
            if item
        ] or ["光线基础良好"]

        return PhotoAnalysis(
            image_id=image_id,
            domain_type=self.domain_type,
            scene_type=f"待视觉模型识别 · {orientation}",
            subjects=SubjectSummary(
                count=1,
                position="主体位置待视觉模型确认",
                face_visibility="待确认",
            ),
            lighting_issues=lighting,
            background_issues=["背景和干扰物待视觉模型确认"],
            portrait_suggestions=[
                "保持主体真实结构",
                "优化主体亮度和清晰度",
                "避免过度生成和细节失真",
            ],
            composition_suggestions=[
                f"保留{orientation}",
                "根据主体位置微调裁切",
                "弱化边缘干扰",
            ],
            recommended_styles=["自然增强", "清晰通透", "氛围风格"],
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
