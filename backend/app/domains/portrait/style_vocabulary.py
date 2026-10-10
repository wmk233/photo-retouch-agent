"""Read-only access to the stylekit style vocabulary.

The AI stylist does not invent styling vocabulary from scratch: it composes
concrete looks out of the curated presets that already ship with the sibling
``stylekit`` package (MIT licensed, treated as read-only here). This module only
*reads* those YAML files and exposes them as typed fragments, so every design can
be traced back to the preset that produced it.

The presets directory is resolved once, in this order:

1. an explicit path passed by the caller,
2. ``PHOTO_AGENT_STYLEKIT_PRESETS_DIR``,
3. a copy vendored inside this repository,
4. the sibling checkout ``<repos>/stylekit/stylekit/presets``.

When none of them exist a small built-in fallback vocabulary keeps the workflow
alive (and keeps the tests independent from the sibling checkout). The fallback
reuses the same preset ids as stylekit, so callers always see the same contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PRESETS_ENV_VAR = "PHOTO_AGENT_STYLEKIT_PRESETS_DIR"

# (bucket, file name, stylekit category)
PRESET_FILES: tuple[tuple[str, str, str], ...] = (
    ("hairstyle_female", "hairstyles_female.yaml", "hairstyle"),
    ("hairstyle_male", "hairstyles_male.yaml", "hairstyle"),
    ("haircolor", "haircolors.yaml", "haircolor"),
    ("makeup", "makeup.yaml", "makeup"),
    ("outfit", "outfits.yaml", "outfit"),
    ("accessory", "accessories.yaml", "accessory"),
)

HAIRSTYLE_BUCKETS: tuple[str, ...] = ("hairstyle_female", "hairstyle_male")

_AGENT_ROOT = Path(__file__).resolve().parents[4]

CANDIDATE_PRESET_DIRS: tuple[Path, ...] = (
    _AGENT_ROOT / "stylekit" / "stylekit" / "presets",
    _AGENT_ROOT.parent / "stylekit" / "stylekit" / "presets",
    _AGENT_ROOT.parent / "stylekit" / "presets",
)

# Length band per stylekit hairstyle preset: the stylist keeps the proposed
# silhouette in a band the person already wears instead of silently inventing a
# different amount of hair.
HAIR_LENGTH_BANDS: dict[str, str] = {
    "air_bangs_long": "long",
    "side_part_waves": "long",
    "beach_waves": "long",
    "braided_low_pony": "long",
    "high_ponytail_bangs": "long",
    "man_bun": "long",
    "french_bangs_bob": "medium",
    "eight_bangs_medium": "medium",
    "lob": "medium",
    "korean_layered": "medium",
    "side_part": "medium",
    "quiff": "medium",
    "pompadour": "medium",
    "curtain_bangs": "medium",
    "korean_perm_short": "short",
    "pixie_cut": "short",
    "textured_crop": "short",
    "undercut": "short",
    "buzz_cut": "short",
    "mullet": "short",
}

LENGTH_BAND_LABELS: dict[str, str] = {
    "short": "短发",
    "medium": "中长发",
    "long": "长发",
    "unknown": "未识别发长",
}


@dataclass(frozen=True)
class StyleFragment:
    """One stylekit preset, normalised for the stylist."""

    preset_id: str
    bucket: str
    category: str
    name_zh: str
    name_en: str
    subcategory: str
    description_zh: str
    description_en: str
    prompt: str
    source_file: str
    best_for: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        """Traceable reference, e.g. ``hairstyles_female.yaml:air_bangs_long``."""
        return f"{self.source_file}:{self.preset_id}"

    @property
    def length_band(self) -> str:
        return HAIR_LENGTH_BANDS.get(self.preset_id, "unknown")


def resolve_presets_dir(explicit: Path | str | None = None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_dir() else None

    env_value = os.getenv(PRESETS_ENV_VAR, "").strip()
    if env_value:
        candidate = Path(env_value)
        if candidate.is_dir():
            return candidate

    for candidate in CANDIDATE_PRESET_DIRS:
        if candidate.is_dir() and any(
            (candidate / file_name).is_file() for _, file_name, _ in PRESET_FILES
        ):
            return candidate
    return None


class StyleVocabulary:
    """Loads the stylekit presets and answers look-up questions about them."""

    def __init__(self, presets_dir: Path | str | None = None) -> None:
        self.presets_dir = resolve_presets_dir(presets_dir)
        self._buckets: dict[str, dict[str, StyleFragment]] = {
            bucket: {} for bucket, _, _ in PRESET_FILES
        }
        self._loaded_files: list[str] = []
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self.presets_dir is not None:
            for bucket, file_name, category in PRESET_FILES:
                path = self.presets_dir / file_name
                if not path.is_file():
                    continue
                self._register_file(bucket, category, file_name, path)
        self._load_fallback()

    def _register_file(
        self,
        bucket: str,
        category: str,
        file_name: str,
        path: Path,
    ) -> None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):  # pragma: no cover - defensive
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            fragment = self._fragment_from_item(bucket, category, file_name, item)
            if fragment is not None:
                self._buckets[bucket][fragment.preset_id] = fragment
        self._loaded_files.append(file_name)

    @staticmethod
    def _fragment_from_item(
        bucket: str,
        category: str,
        file_name: str,
        item: object,
    ) -> StyleFragment | None:
        if not isinstance(item, dict):
            return None
        preset_id = str(item.get("id") or "").strip()
        if not preset_id:
            return None
        best_for = item.get("best_for")
        return StyleFragment(
            preset_id=preset_id,
            bucket=bucket,
            category=category,
            name_zh=str(item.get("name_zh") or preset_id),
            name_en=str(item.get("name_en") or preset_id),
            subcategory=str(item.get("subcategory") or ""),
            description_zh=str(item.get("description_zh") or ""),
            description_en=str(item.get("description_en") or ""),
            prompt=str(item.get("prompt") or "").strip(),
            source_file=file_name,
            best_for=best_for if isinstance(best_for, dict) else {},
        )

    def _load_fallback(self) -> None:
        """Guarantee a usable vocabulary even without the stylekit checkout."""
        for fragment in _fallback_fragments().values():
            self._buckets[fragment.bucket].setdefault(fragment.preset_id, fragment)

    # ---------------------------------------------------------------- accessors
    @property
    def available(self) -> bool:
        """True when the real stylekit preset files were found and parsed."""
        return bool(self._loaded_files)

    @property
    def loaded_files(self) -> tuple[str, ...]:
        return tuple(self._loaded_files)

    @property
    def source_label(self) -> str:
        if self.available and self.presets_dir is not None:
            return str(self.presets_dir)
        return "built-in fallback"

    def get(self, bucket: str, preset_id: str) -> StyleFragment | None:
        return self._buckets.get(bucket, {}).get(preset_id)

    def find_hair(self, preset_id: str) -> StyleFragment | None:
        for bucket in HAIRSTYLE_BUCKETS:
            fragment = self.get(bucket, preset_id)
            if fragment is not None:
                return fragment
        return None

    def list(self, bucket: str) -> tuple[StyleFragment, ...]:
        return tuple(self._buckets.get(bucket, {}).values())

    def first(self, bucket: str, preset_ids: tuple[str, ...]) -> StyleFragment | None:
        for preset_id in preset_ids:
            fragment = self.get(bucket, preset_id)
            if fragment is not None:
                return fragment
        return None


def _fallback(bucket: str, category: str, file_name: str, **fields: str) -> StyleFragment:
    return StyleFragment(
        bucket=bucket,
        category=category,
        source_file=file_name,
        best_for={},
        **fields,
    )


def _fallback_fragments() -> dict[str, StyleFragment]:
    """A minimal, id-compatible subset of the stylekit vocabulary."""
    hairstyles = {
        "air_bangs_long": "空气刘海+长发微卷",
        "lob": "Lob 长锁骨发",
        "pixie_cut": "精灵短发",
        "side_part_waves": "侧分蓬松大波浪",
    }
    haircolors = {
        "milk_tea_brown": "奶茶棕",
        "chocolate_brown": "巧克力棕",
        "misty_ash_brown": "雾感灰棕",
        "burgundy_brown": "酒红棕",
    }
    makeups = {
        "clean_nude": "干净裸妆",
        "peach_sweet": "蜜桃甜系",
        "korean_dewy": "韩系水光妆",
    }
    outfits = {
        "knit_sweater": "针织毛衣休闲",
        "oversized_streetwear": "宽松街头",
        "vintage_french": "法式复古",
        "techwear": "机能风",
    }
    accessories = {
        "pearl_stud_earrings": "珍珠耳钉",
        "black_baseball_cap": "黑色棒球帽",
        "gold_hoop_earrings": "金色圈形耳环",
        "round_metal_glasses": "圆形金属眼镜",
    }

    fragments: dict[str, StyleFragment] = {}
    for preset_id, name_zh in hairstyles.items():
        fragments[f"hairstyle_female:{preset_id}"] = _fallback(
            "hairstyle_female",
            "hairstyle",
            "hairstyles_female.yaml",
            preset_id=preset_id,
            name_zh=name_zh,
            name_en=preset_id,
            subcategory="fallback",
            description_zh=name_zh,
            description_en=preset_id,
            prompt=f"{name_zh} ({preset_id}) hairstyle, styled naturally.",
        )
    for bucket, file_name, items in (
        ("haircolor", "haircolors.yaml", haircolors),
        ("makeup", "makeup.yaml", makeups),
        ("outfit", "outfits.yaml", outfits),
        ("accessory", "accessories.yaml", accessories),
    ):
        category = bucket
        for preset_id, name_zh in items.items():
            fragments[f"{bucket}:{preset_id}"] = _fallback(
                bucket,
                category,
                file_name,
                preset_id=preset_id,
                name_zh=name_zh,
                name_en=preset_id,
                subcategory="fallback",
                description_zh=name_zh,
                description_en=preset_id,
                prompt=f"{name_zh} ({preset_id}).",
            )

    return fragments


def fallback_fragment(bucket: str, preset_id: str) -> StyleFragment | None:
    """Return one built-in fallback fragment without loading any YAML."""
    return _fallback_fragments().get(f"{bucket}:{preset_id}")
