from __future__ import annotations
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class PromptBrainError(RuntimeError):
    """Provider error whose message never contains API credentials."""


class OpenAICompatiblePromptBrain:
    def __init__(
        self,
        provider_name: str,
        api_key: str,
        endpoint: str,
        model_name: str,
        vision_mode: str = "derived",
        image_url_mode: str = "data_url",
        timeout_seconds: float = 180,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if vision_mode not in {"direct", "derived"}:
            raise ValueError("vision_mode must be direct or derived")
        if image_url_mode not in {"data_url", "raw_base64"}:
            raise ValueError("image_url_mode must be data_url or raw_base64")

        self.provider_name = provider_name
        self.model_name = model_name
        self.vision_mode = vision_mode
        self._api_key = api_key
        self.endpoint = endpoint
        self.image_url_mode = image_url_mode
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def analyze(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        baseline_payload = baseline.model_dump(by_alias=True)
        system_prompt = (
            "你是照片美化 Agent 的视觉分析大脑。"
            "请识别照片主体、场景、光线、背景、构图和可优化点，"
            "覆盖人像、风景、街拍、室内、商品等照片。"
            "保持客观，不推断敏感身份属性。"
            "只返回 JSON 对象，字段为 domainType、sceneType、subjects、lightingIssues、"
            "backgroundIssues、portraitSuggestions、compositionSuggestions、"
            "recommendedStyles、riskFlags。domainType 只能是 portrait、landscape、"
            "product、general 之一。subjects 必须是单个 JSON 对象而不是数组，"
            "且只包含 count、position、faceVisibility 三个字段。"
        )
        user_text = json.dumps(
            {
                "baselineAnalysis": baseline_payload,
                "instruction": (
                    "直接观察随消息提供的图片并修正基线分析。"
                    if self.vision_mode == "direct"
                    else "基于系统提供的结构化图像统计分析进行规划；不要声称直接看到了图片。"
                ),
            },
            ensure_ascii=False,
        )
        content = self._complete(
            system_prompt,
            user_text,
            image_path if self.vision_mode == "direct" else None,
        )
        enriched = self._parse_json_object(content)
        enriched["subjects"] = self._normalize_subjects(
            enriched.get("subjects"),
            baseline_payload["subjects"],
        )
        domain_type = str(enriched.get("domainType") or baseline.domain_type).lower()
        if domain_type not in {"portrait", "landscape", "product", "general"}:
            domain_type = "general"
        merged = {
            **baseline_payload,
            **enriched,
            "imageId": image_id,
            "domainType": domain_type,
            "brainProvider": self.provider_name,
            "brainModel": self.model_name,
            "visionMode": self.vision_mode,
        }
        try:
            return PhotoAnalysis.model_validate(merged)
        except ValueError as exc:
            raise PromptBrainError(
                f"{self.provider_name} returned an invalid photo analysis."
            ) from exc

    @staticmethod
    def _normalize_subjects(
        subjects: Any,
        baseline_subjects: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(subjects, dict):
            return {**baseline_subjects, **subjects}

        if isinstance(subjects, list):
            subject_items = [item for item in subjects if isinstance(item, dict)]
            first_subject = subject_items[0] if subject_items else {}
            return {
                "count": len(subject_items) or baseline_subjects["count"],
                "position": (
                    first_subject.get("position")
                    or baseline_subjects["position"]
                ),
                "faceVisibility": (
                    first_subject.get("faceVisibility")
                    or baseline_subjects["faceVisibility"]
                ),
            }

        return baseline_subjects

    def optimize(
        self,
        source_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> RetouchPlan:
        system_prompt = (
            "你是专业照片修图 Agent 的提示词规划器。"
            "请结合照片内容与输入方案，生成可直接交给图像编辑模型的中文指令。"
            "人像必须保持身份、五官比例和真实皮肤纹理；"
            "风景和其他照片必须保持主体结构与真实空间关系。"
            "只返回 JSON 对象，字段为 editPrompt、negativePrompt、expectedChanges。"
        )
        user_text = json.dumps(
            {
                "plan": plan.model_dump(by_alias=True),
                "userInstruction": user_instruction,
                "visionNotice": (
                    "请直接观察随消息提供的原图。"
                    if self.vision_mode == "direct"
                    else "当前模型无图片输入能力，请依据结构化方案规划，不要声称直接看图。"
                ),
            },
            ensure_ascii=False,
        )
        content = self._complete(
            system_prompt,
            user_text,
            source_path if self.vision_mode == "direct" else None,
        )
        optimized = self._parse_json_object(content)
        edit_prompt = str(optimized.get("editPrompt") or "").strip()
        if not edit_prompt:
            raise PromptBrainError(
                f"{self.provider_name} response did not contain editPrompt."
            )

        expected_changes = optimized.get("expectedChanges")
        if not isinstance(expected_changes, list):
            expected_changes = plan.expected_changes

        return plan.model_copy(
            update={
                "edit_prompt": edit_prompt,
                "negative_prompt": str(
                    optimized.get("negativePrompt") or plan.negative_prompt
                ).strip(),
                "expected_changes": [str(item) for item in expected_changes],
            }
        )

    def _complete(
        self,
        system_prompt: str,
        user_text: str,
        image_path: Path | None,
    ) -> str:
        user_content: str | list[dict[str, Any]] = user_text
        if image_path is not None:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(image_path)},
                },
                {"type": "text", "text": user_text},
            ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "stream": False,
        }

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                response = client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise PromptBrainError(
                f"{self.provider_name} API request failed."
            ) from exc
        return self._extract_content(response)

    def _encode_image(self, path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise PromptBrainError("Unsupported image type for visual analysis.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        if self.image_url_mode == "raw_base64":
            return encoded
        return f"data:{mime_type};base64,{encoded}"

    def _extract_content(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError as exc:
            raise PromptBrainError(
                f"{self.provider_name} returned invalid JSON (HTTP {response.status_code})."
            ) from exc

        if response.is_error or payload.get("error"):
            error = payload.get("error") or {}
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            ) or payload.get("message") or "Request failed."
            safe_message = str(message).replace(self._api_key, "[redacted]")
            raise PromptBrainError(
                f"{self.provider_name} API call failed "
                f"(HTTP {response.status_code}): {safe_message}"
            )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptBrainError(
                f"{self.provider_name} response did not contain message content."
            ) from exc

        if isinstance(content, list):
            text_parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("text")
            ]
            return "\n".join(text_parts)
        return str(content)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise PromptBrainError("Brain response was not a JSON object.")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PromptBrainError("Brain response contained invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise PromptBrainError("Brain response was not a JSON object.")
        return parsed
