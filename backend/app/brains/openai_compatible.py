import json
from typing import Any

import httpx

from app.schemas.retouch import RetouchPlan


class PromptBrainError(RuntimeError):
    """Safe brain-provider error that never contains API credentials."""


class OpenAICompatiblePromptBrain:
    def __init__(
        self,
        provider_name: str,
        api_key: str,
        endpoint: str,
        model_name: str,
        timeout_seconds: float = 180,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def optimize(self, plan: RetouchPlan, user_instruction: str) -> RetouchPlan:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是专业人像修图 Agent 的提示词规划器。"
                        "请基于输入方案生成可直接交给图像编辑模型的中文指令。"
                        "必须保持人物身份、五官比例和真实皮肤纹理。"
                        "只返回 JSON 对象，字段为 editPrompt、negativePrompt、expectedChanges。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "plan": plan.model_dump(by_alias=True),
                            "userInstruction": user_instruction,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
            "stream": False,
        }

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
        content = self._extract_content(response)
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
                error.get("message")
                if isinstance(error, dict)
                else str(error)
            ) or payload.get("message") or "Request failed."
            raise PromptBrainError(
                f"{self.provider_name} API call failed (HTTP {response.status_code}): {message}"
            )
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptBrainError(
                f"{self.provider_name} response did not contain message content."
            ) from exc

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
