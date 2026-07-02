from __future__ import annotations
import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.schemas.retouch import RetouchPlan


class QwenProviderError(RuntimeError):
    """Safe provider error that never contains API credentials."""


class QwenImageProvider:
    output_extension = ".png"

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model_name: str,
        provider_name: str = "qwen",
        parameter_profile: str = "qwen",
        timeout_seconds: float = 180,
        download_limit_bytes: int = 30 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if parameter_profile not in {"qwen", "wan"}:
            raise ValueError("parameter_profile must be qwen or wan")
        self.provider_name = provider_name
        self._api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        self.parameter_profile = parameter_profile
        self.timeout_seconds = timeout_seconds
        self.download_limit_bytes = download_limit_bytes
        self._transport = transport

    def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> None:
        parameters: dict[str, Any] = {
            "n": 1,
            "watermark": False,
        }
        if self.parameter_profile == "qwen":
            parameters.update(
                {
                    "negative_prompt": plan.negative_prompt,
                    "prompt_extend": True,
                }
            )
        else:
            parameters["size"] = "2K"

        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": self._encode_image(source_path)},
                            {"text": self._build_prompt(plan, user_instruction)},
                        ],
                    }
                ]
            },
            "parameters": parameters,
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
            result = self._parse_response(response)
            image_url = self._extract_image_url(result)
            image_response = client.get(image_url)
            self._save_download(image_response, output_path)

    @staticmethod
    def _encode_image(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise QwenProviderError("Unsupported input image type.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        instructions = [
            plan.edit_prompt,
            "保持人物身份、五官比例和真实皮肤纹理，不改变本人特征。",
            "结果应自然、真实，避免过度磨皮、脸部畸变和明显 AI 生成痕迹。",
        ]
        if user_instruction.strip():
            instructions.append(f"用户补充要求：{user_instruction.strip()}")
        return "\n".join(instructions)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise QwenProviderError(
                f"DashScope API returned invalid JSON (HTTP {response.status_code})."
            ) from exc

        if response.is_error or payload.get("code"):
            code = payload.get("code") or f"HTTP {response.status_code}"
            message = payload.get("message") or "Request failed."
            raise QwenProviderError(f"DashScope API call failed ({code}): {message}")
        return payload

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        try:
            contents = payload["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenProviderError(
                "DashScope API response did not contain an output image."
            ) from exc

        for item in contents:
            image_url = item.get("image") if isinstance(item, dict) else None
            if image_url:
                return str(image_url)
        raise QwenProviderError("DashScope API response did not contain an output image.")

    def _save_download(self, response: httpx.Response, output_path: Path) -> None:
        if response.is_error:
            raise QwenProviderError(
                f"Failed to download DashScope output image (HTTP {response.status_code})."
            )
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise QwenProviderError("DashScope output URL did not return an image.")
        if len(response.content) > self.download_limit_bytes:
            raise QwenProviderError("DashScope output image exceeded the download limit.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
