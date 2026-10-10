"""The AI styling planner: one vision request, three complete looks.

This is the brain that makes the styling workflow genuinely *AI* authored
instead of a fixed ``_Axis`` template. It reuses the very same
``OpenAICompatiblePromptBrain`` transport, image encoding and JSON parsing as the
analysis brain, so it adds no new model, endpoint or dependency.

Design contract:

* It receives the locally measured, observable ``FaceFacts`` plus the full
  catalogue of available stylekit presets, and (when the provider supports
  vision) the original photo itself.
* It returns *three* mutually distinct proposals in a single response, so a
  design pass costs exactly one model request.
* It never invents vocabulary: every chosen preset id must exist in the
  catalogue that was handed to it, otherwise the whole design fails loudly
  instead of silently falling back to a template.

Private-attribute inference (gender, age, ethnicity, occupation, personality) is
explicitly forbidden in the system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.brains.openai_compatible import (
    OpenAICompatiblePromptBrain,
    PromptBrainError,
)

# Catalogue bucket -> the key the model fills inside ``selectedPresets``. The
# hairstyle bucket is called ``hairstyle`` in the catalogue but ``hair`` in the
# answer contract, so the two must be mapped explicitly.
PRESET_BUCKETS: tuple[str, ...] = (
    "hairstyle",
    "haircolor",
    "makeup",
    "outfit",
    "accessory",
)

_SELECTION_KEYS: dict[str, str] = {
    "hairstyle": "hair",
    "haircolor": "haircolor",
    "makeup": "makeup",
    "outfit": "outfit",
    "accessory": "accessory",
}

# The response contract the planner asks the model to fill in. Kept next to the
# validator so the two can never drift apart.
_RESPONSE_CONTRACT: dict[str, Any] = {
    "proposals": [
        {
            "slug": "ascii_slug_for_the_look",
            "name": "中文造型名称",
            "designDirection": "一句话风格方向",
            "designReason": "为什么这样设计（基于可见特征）",
            "whyItFits": "为什么适合照片里的这个人",
            "intensity": "mild | medium | bold",
            "hair": {
                "length": "长度",
                "structure": "结构",
                "bangs": "刘海",
                "layers": "层次",
                "color": "发色方向",
            },
            "makeup": {
                "brows": "眉毛",
                "eyes": "眼妆",
                "lips": "唇妆",
                "base": "底妆",
            },
            "outfit": {
                "style": "款式",
                "material": "材质",
                "color": "颜色",
                "pairing": "搭配",
            },
            "accessory": "配饰说明",
            "palette": "整体配色（中文）",
            "identityAnchors": ["需要保留的可见辨识特征"],
            "editPrompt": "给图像编辑模型的具体改造指令（中英不限，要具体）",
            "negativePrompt": ["要避免的效果"],
            "selectedPresets": {
                "hair": "hairstyle presetId",
                "haircolor": "haircolor presetId",
                "makeup": "makeup presetId",
                "outfit": "outfit presetId",
                "accessory": "accessory presetId",
            },
        }
    ]
}

# The prompt differs by transport: a "direct" brain is handed the pixels and may
# describe them, while a "derived" brain (DeepSeek and other text-only models)
# only ever receives locally measured facts and must never claim to have seen
# the photo.
_SYSTEM_PROMPT_HEAD_VISION = (
    "你是「AI Beauty Style Lab」的资深整体形象设计总监（AI Stylist）。"
    "你会看到一位真人的照片、由本地程序测量得到的可见外形事实（faceFacts），"
    "以及一份可用的风格库预设清单（stylePresetCatalogue）。"
    "请为这一个人设计三套完整、彼此差异明显的整体形象改造方案。\n"
)

_SYSTEM_PROMPT_HEAD_DERIVED = (
    "你是「AI Beauty Style Lab」的资深整体形象设计总监（AI Stylist）。"
    "重要：你并没有看到这位真人的原图（本次没有图像输入）。"
    "你只会收到由本地程序测量得到的可见外形事实（faceFacts）"
    "以及一份可用的风格库预设清单（stylePresetCatalogue）。"
    "请只依据这些结构化事实，为这一个人设计三套完整、彼此差异明显的整体形象改造方案，"
    "不要描述任何照片细节，也不要声称自己看过照片。\n"
)

_SYSTEM_PROMPT_RULES = (
    "硬性要求：\n"
    "1. 只依据给定的可观察视觉信息与 faceFacts 判断；"
    "绝不推断或描述人物的性别、年龄、职业、性格、民族、宗教信仰等无法从照片确认的属性。\n"
    "2. 三套方案必须在发型、发色、妆容、服装搭配上有实质区别，"
    "不能只是换三个形容词，也不要是同一种造型的三个变体。\n"
    "3. 每套方案必须从 stylePresetCatalogue 中分别挑选 hair、haircolor、makeup、"
    "outfit、accessory 各一个预设；presetId 必须真实存在于清单中，不得编造；"
    "不得输出清单以外的 id。\n"
    "4. 每套方案必须给出 identityAnchors：需要保留的、从照片可观察到的辨识特征"
    "（脸型轮廓、五官相对比例、肤色基调、是否佩戴眼镜、发际线等）。\n"
    "5. 保脸是尽力而为，绝不承诺百分之百保脸，也不要声称能保证身份完全不变。\n"
    "6. 只返回一个 JSON 对象，字段严格符合 requiredResponse 的结构，不要输出任何多余文字。"
)


def _styling_system_prompt(vision_mode: str) -> str:
    head = (
        _SYSTEM_PROMPT_HEAD_VISION
        if vision_mode == "direct"
        else _SYSTEM_PROMPT_HEAD_DERIVED
    )
    return head + _SYSTEM_PROMPT_RULES


class AIStylingPlanner(OpenAICompatiblePromptBrain):
    """Authors three personalised looks with a real vision model."""

    async def plan(
        self,
        image_path: Path | None,
        face_facts: dict[str, Any],
        preset_catalogue: dict[str, list[dict[str, Any]]],
        valid_preset_ids: dict[str, set[str]],
        identity_anchors: tuple[str, ...] = (),
        user_instruction: str = "",
        analysis_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one design pass and return ``{"proposals": [...]}``."""
        user_text = self._build_user_text(
            face_facts=face_facts,
            preset_catalogue=preset_catalogue,
            identity_anchors=identity_anchors,
            user_instruction=user_instruction,
            analysis_summary=analysis_summary or {},
            vision_mode=self.vision_mode,
        )
        content = await self._complete(
            _styling_system_prompt(self.vision_mode),
            user_text,
            image_path if self.vision_mode == "direct" else None,
            temperature=0.7,
        )
        payload = self._parse_json_object(content)
        return self._validate(payload, valid_preset_ids)

    # ------------------------------------------------------------- prompt build
    @staticmethod
    def _build_user_text(
        face_facts: dict[str, Any],
        preset_catalogue: dict[str, list[dict[str, Any]]],
        identity_anchors: tuple[str, ...],
        user_instruction: str,
        analysis_summary: dict[str, Any],
        vision_mode: str = "derived",
    ) -> str:
        catalogue = {
            bucket: [
                {
                    "presetId": item.get("presetId"),
                    "nameZh": item.get("nameZh"),
                    "descriptionZh": item.get("descriptionZh"),
                    "bestFor": item.get("bestFor") or {},
                }
                for item in entries
            ]
            for bucket, entries in preset_catalogue.items()
        }

        return json.dumps(
            {
                "task": (
                    "为照片里的这个人设计三套完整形象改造方案。"
                    + (
                        "直接观察随消息提供的原图（若提供），结合 faceFacts 做设计判断。"
                        if vision_mode == "direct"
                        else "本次没有图像输入：请只依据本地程序测量的 faceFacts "
                        "与结构化分析做设计，不要声称自己看过原图。"
                    )
                ),
                "visionMode": vision_mode,
                "faceFacts": face_facts,
                "analysisSummary": analysis_summary,
                "identityAnchorsToKeep": list(identity_anchors),
                "stylePresetCatalogue": catalogue,
                "userRequest": (user_instruction or "").strip(),
                "requiredResponse": _RESPONSE_CONTRACT,
            },
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------- validation
    @staticmethod
    def _validate(
        payload: dict[str, Any],
        valid_preset_ids: dict[str, set[str]],
    ) -> dict[str, Any]:
        proposals = payload.get("proposals")
        if not isinstance(proposals, list) or len(proposals) < 3:
            raise PromptBrainError(
                "The styling model did not return three complete proposals."
            )
        proposals = proposals[:3]

        names: set[str] = set()
        signatures: set[tuple[str, ...]] = set()
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                raise PromptBrainError(
                    "The styling model returned a malformed proposal."
                )
            name = str(proposal.get("name") or "").strip()
            if not name:
                raise PromptBrainError(
                    "The styling model returned a proposal without a name."
                )
            names.add(name)

            selection = proposal.get("selectedPresets")
            if not isinstance(selection, dict):
                raise PromptBrainError(
                    f"Look 「{name}」 is missing its stylekit preset selection."
                )
            resolved: list[str] = []
            for bucket in PRESET_BUCKETS:
                selection_key = _SELECTION_KEYS[bucket]
                preset_id = str(selection.get(selection_key) or "").strip()
                if not preset_id:
                    raise PromptBrainError(
                        f"Look 「{name}」 did not pick a {bucket} preset."
                    )
                if preset_id not in valid_preset_ids.get(bucket, set()):
                    raise PromptBrainError(
                        f"Look 「{name}」 picked an unknown {bucket} preset: "
                        f"{preset_id}"
                    )
                resolved.append(preset_id)
            signatures.add(tuple(resolved))

            if not str(proposal.get("editPrompt") or "").strip():
                raise PromptBrainError(
                    f"Look 「{name}」 is missing its image edit instruction."
                )
            for key in ("hair", "makeup", "outfit"):
                if not isinstance(proposal.get(key), dict) or not proposal.get(key):
                    raise PromptBrainError(
                        f"Look 「{name}」 is missing its {key} design."
                    )

        if len(names) < 3:
            raise PromptBrainError(
                "The styling model returned duplicate look names; the three "
                "designs must be distinct."
            )
        if len(signatures) < 3:
            raise PromptBrainError(
                "The styling model reused the same stylekit presets across looks; "
                "the three designs must be distinct."
            )
        return {"proposals": proposals}
