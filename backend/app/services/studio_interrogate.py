"""反推提示词：把一张图交给视觉模型，换回一段能直接出图的话。

这是生图工作台的辅助入口，不是美颜那条管线。美颜管线的规矩是「本地量出来的
事实不出机器」；反推恰恰要把画面交给云端视觉模型，所以它只属于工作台：这里
唯一被保护的东西是钱，而不是画面。

规矩和生图完全一致，而且是同一条：先报价、后花钱。``quote`` 不联网、不花钱，
只回答「这一步要付几次请求」；``interrogate`` 必须收到 ``confirmPaid`` 才会真的
发出那一次请求。反推是独立的一次付费请求，不并入出图，这样账单上永远能对上。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.brains.openai_compatible import (
    OpenAICompatiblePromptBrain,
    PromptBrainError,
)
from app.core.config import Settings, settings as default_settings
from app.core.errors import bad_request
from app.services.storage import StorageService

logger = logging.getLogger("photo_retouch.studio.interrogate")

FOCUS_REPLICATE = "replicate"
FOCUS_STYLE = "style"
FOCUSES = (FOCUS_REPLICATE, FOCUS_STYLE)

#: One interrogation is one billed call. Not a batch, so there is nothing to
#: clamp -- it is a constant, and every price shown to the user reads from here.
PAID_REQUESTS_PER_INTERROGATION = 1

MAX_ALTERNATIVES = 3


@dataclass(frozen=True)
class InterrogateModel:
    model_id: str
    label: str
    note: str

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.model_id, "label": self.label, "note": self.note}


#: Vision-language models that can actually see the picture. Kept apart from
#: ``SILICONFLOW_VISION_MODEL`` on purpose: that setting defaults to a text-only
#: model, and a text-only model would happily invent a description of a photo it
#: never received.
INTERROGATE_MODELS: tuple[InterrogateModel, ...] = (
    InterrogateModel(
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        label="Qwen3-VL 32B",
        note="默认：看图画质最好，写出来的提示词最具体",
    ),
    InterrogateModel(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        label="Qwen3-VL 30B-A3B",
        note="MoE 版，速度更快，质量接近 32B",
    ),
    InterrogateModel(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        label="Qwen3-VL 8B",
        note="最便宜，适合快速试写",
    ),
)

DEFAULT_INTERROGATE_MODEL = INTERROGATE_MODELS[0].model_id

_INTERROGATE_INDEX = {model.model_id: model for model in INTERROGATE_MODELS}

_SHARED_RULES = (
    "你是资深摄影指导，负责把一张照片翻译成可以直接送进文生图模型的提示词。"
    "只写你在画面里真实看到的东西，不要猜测拍摄地点、人物身份或相机型号。"
    "可以描述发型、妆容、衣着、姿态、气质这些看得见的特征；"
    "不要推断年龄、职业、民族、国籍、宗教、健康状况等身份属性。"
    "主提示词用中文写成一整段自然语言，按「主体与动作 → 环境与背景 → 光线与时间 → "
    "构图与镜头（视角、焦距、景深）→ 色调、质感与风格」的顺序展开，"
    "用具体、可执行的视觉词汇，不要写「很美」「高级感」这类空话，"
    "也不要出现「不要」「去掉」这类否定句（该排除的内容写进 negativePrompt）。"
    "只返回 JSON 对象，字段为 prompt、negativePrompt、alternatives。"
    "prompt 是主提示词；negativePrompt 是生成时建议排除的内容；"
    "alternatives 是 2 到 3 条与主提示词同义但侧重不同的备选说法。"
    "三个字段都是字符串或字符串数组，不要附加解释文字。"
)

_FOCUS_INSTRUCTIONS = {
    FOCUS_REPLICATE: (
        "本次目标是尽量复刻这张画面：主体是什么、在做什么、处在什么环境、"
        "什么光线、什么构图，都要写清楚。"
    ),
    FOCUS_STYLE: (
        "本次目标只要风格，不要复刻画面内容：只写光线、色调、影调、质感、"
        "气质和整体氛围，不要说出画面里具体的主体是什么。"
    ),
}


def _system_prompt(focus: str) -> str:
    return f"{_SHARED_RULES}{_FOCUS_INSTRUCTIONS.get(focus, _FOCUS_INSTRUCTIONS[FOCUS_REPLICATE])}"


def _user_text(focus: str, hint: str) -> str:
    payload: dict[str, Any] = {"focus": focus, "instruction": "观察随消息提供的图片并返回 JSON。"}
    if hint:
        payload["extraRequirement"] = hint
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class StudioInterrogateRequest:
    source_image_id: str
    model_id: str | None = None
    focus: str = FOCUS_REPLICATE
    hint: str = ""
    confirm_paid: bool = False


class StudioInterrogateService:
    """Owns the reverse-prompt catalogue, its quote and its single paid call."""

    def __init__(
        self,
        config: Settings = default_settings,
        storage: StorageService | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or StorageService(config)
        # Only the tests inject a transport; production always uses the network.
        self.transport = transport

    # --- catalogue --------------------------------------------------------

    def catalogue(self) -> dict[str, Any]:
        return {
            "models": [model.to_payload() for model in INTERROGATE_MODELS],
            "defaultModel": DEFAULT_INTERROGATE_MODEL,
        }

    # --- quoting ----------------------------------------------------------

    def quote(self, request: StudioInterrogateRequest) -> dict[str, Any]:
        """Resolve a request into the exact number of paid calls. Never online."""
        source_image_id = (request.source_image_id or "").strip()
        if not source_image_id:
            raise bad_request("反推提示词要有一张看图的对象，请先上传一张输入图。")
        if self.storage.resolve_image_path(source_image_id) is None:
            raise bad_request("找不到这张输入图，请重新上传。")

        model = self._resolve_model(request)
        focus = request.focus if request.focus in FOCUSES else FOCUS_REPLICATE
        return {
            "model": model.model_id,
            "modelLabel": model.label,
            "focus": focus,
            "sourceImageId": source_image_id,
            "paidRequests": PAID_REQUESTS_PER_INTERROGATION,
            "requiresConfirmation": True,
            "note": f"本次将发起 1 次真实付费请求（{model.label} 看图写提示词）。",
        }

    def _resolve_model(self, request: StudioInterrogateRequest) -> InterrogateModel:
        requested = (request.model_id or "").strip() or DEFAULT_INTERROGATE_MODEL
        model = _INTERROGATE_INDEX.get(requested)
        if model is None:
            raise bad_request(f"这个反推模型还没有接入：{requested}")
        return model

    # --- the paid call ----------------------------------------------------

    async def interrogate(self, request: StudioInterrogateRequest) -> dict[str, Any]:
        """Run the quoted interrogation. Never retries: a failed call is reported."""
        if not request.confirm_paid:
            raise bad_request("未确认费用，已拒绝发起付费请求。")
        if not self.config.siliconflow_api_key:
            raise bad_request(
                "还没有配置 SILICONFLOW_API_KEY，无法反推提示词。"
                "在仓库根目录的 .env 里写入后重启服务即可。"
            )

        quote = self.quote(request)
        model = _INTERROGATE_INDEX[quote["model"]]
        source_path = self.storage.resolve_image_path(quote["sourceImageId"])
        if source_path is None:
            raise bad_request("找不到这张输入图，请重新上传。")

        brain = OpenAICompatiblePromptBrain(
            provider_name="siliconflow",
            api_key=self.config.siliconflow_api_key,
            endpoint=f"{self.config.siliconflow_base_url}/chat/completions",
            model_name=model.model_id,
            vision_mode="direct",
            timeout_seconds=self.config.provider_timeout_seconds,
            transport=self.transport,
        )
        try:
            payload = await brain.complete_vision_json(
                system_prompt=_system_prompt(quote["focus"]),
                user_text=_user_text(quote["focus"], (request.hint or "").strip()),
                image_path=source_path,
                temperature=0.3,
            )
        except PromptBrainError as exc:
            # One paid call, reported and never repeated.
            logger.warning("studio interrogate failed: %s", exc)
            raise bad_request(f"反推失败：{exc}") from exc

        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise bad_request("模型这次没有返回可用的提示词，换个模型或稍后再试。")

        return {
            "quote": quote,
            "prompt": prompt,
            "negativePrompt": str(payload.get("negativePrompt") or "").strip(),
            "alternatives": self._alternatives(payload.get("alternatives"), prompt),
            "model": model.model_id,
            "modelLabel": model.label,
            "focus": quote["focus"],
            "sourceImageId": quote["sourceImageId"],
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "paidRequests": PAID_REQUESTS_PER_INTERROGATION,
        }

    @staticmethod
    def _alternatives(raw: Any, prompt: str) -> list[str]:
        if not isinstance(raw, list):
            return []
        alternatives: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text and text != prompt and text not in alternatives:
                alternatives.append(text)
        return alternatives[:MAX_ALTERNATIVES]


__all__ = [
    "DEFAULT_INTERROGATE_MODEL",
    "FOCUS_REPLICATE",
    "FOCUS_STYLE",
    "INTERROGATE_MODELS",
    "InterrogateModel",
    "PAID_REQUESTS_PER_INTERROGATION",
    "StudioInterrogateRequest",
    "StudioInterrogateService",
]
