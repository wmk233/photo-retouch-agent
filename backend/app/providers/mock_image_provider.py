from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.schemas.retouch import RetouchPlan


class MockImageProvider:
    provider_name = "local"
    model_name = "pillow-mock-retoucher"
    output_extension = ".jpg"

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> None:
        image = Image.open(source_path).convert("RGB")
        recipe = self._recipe(plan, user_instruction)

        if recipe["crop_square"]:
            image = self._center_crop_square(image)
        elif max(image.size) > 1500:
            image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)

        if recipe["background_soften"]:
            softened = image.filter(ImageFilter.GaussianBlur(radius=1.2))
            image = Image.blend(softened, image, 0.86)

        image = ImageOps.autocontrast(image, cutoff=recipe["cutoff"])
        image = ImageEnhance.Brightness(image).enhance(recipe["brightness"])
        image = ImageEnhance.Contrast(image).enhance(recipe["contrast"])
        image = ImageEnhance.Color(image).enhance(recipe["color"])

        if recipe["warmth"] != 0:
            image = self._apply_warmth(image, recipe["warmth"])
        if recipe["vignette"]:
            image = self._apply_vignette(image, recipe["vignette"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=92, optimize=True)

    def _recipe(self, plan: RetouchPlan, instruction: str) -> dict[str, float | bool | int]:
        text = instruction.lower()
        wants_brighter = any(word in text for word in ["亮", "提亮", "白", "bright"])
        wants_natural = any(word in text for word in ["自然", "轻", "别太", "不要太", "natural"])
        wants_avatar = any(word in text for word in ["头像", "裁", "简历", "avatar"])

        recipes: dict[str, dict[str, float | bool | int]] = {
            "natural": {
                "brightness": 1.07,
                "contrast": 1.04,
                "color": 1.06,
                "warmth": 0.04,
                "vignette": 0.02,
                "cutoff": 1,
                "crop_square": False,
                "background_soften": False,
            },
            "avatar": {
                "brightness": 1.1,
                "contrast": 1.08,
                "color": 1.04,
                "warmth": 0.03,
                "vignette": 0.02,
                "cutoff": 1,
                "crop_square": True,
                "background_soften": True,
            },
            "mood": {
                "brightness": 1.02,
                "contrast": 1.14,
                "color": 1.18,
                "warmth": 0.08,
                "vignette": 0.08,
                "cutoff": 2,
                "crop_square": False,
                "background_soften": False,
            },
        }
        recipe = dict(recipes.get(plan.plan_id, recipes["natural"]))

        if wants_brighter:
            recipe["brightness"] = float(recipe["brightness"]) + (0.04 if wants_natural else 0.1)
        if wants_natural:
            recipe["contrast"] = min(float(recipe["contrast"]), 1.06)
            recipe["color"] = min(float(recipe["color"]), 1.08)
            recipe["vignette"] = min(float(recipe["vignette"]), 0.03)
        if wants_avatar:
            recipe["crop_square"] = True

        return recipe

    @staticmethod
    def _center_crop_square(image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        cropped = image.crop((left, top, left + side, top + side))
        return cropped.resize((1024, 1024), Image.Resampling.LANCZOS)

    @staticmethod
    def _apply_warmth(image: Image.Image, amount: float) -> Image.Image:
        red, green, blue = image.split()
        red = ImageEnhance.Brightness(red).enhance(1 + amount)
        blue = ImageEnhance.Brightness(blue).enhance(1 - amount * 0.55)
        return Image.merge("RGB", (red, green, blue))

    @staticmethod
    def _apply_vignette(image: Image.Image, amount: float) -> Image.Image:
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        center = Image.new("L", (max(width, height), max(width, height)), 255)
        center = center.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.26))
        left = (center.width - width) // 2
        top = (center.height - height) // 2
        mask.paste(center.crop((left, top, left + width, top + height)))
        darkened = ImageEnhance.Brightness(image).enhance(1 - amount)
        return Image.composite(image, darkened, mask)
