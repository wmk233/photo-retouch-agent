from __future__ import annotations
import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.schemas.retouch import RetouchPlan


class SeedreamProviderError(RuntimeError):
    """Safe Seedream error that never contains API credentials."""


class SeedreamImageProvider:
    provider_name = "seedream"
    output_extension = ".png"

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model_name: str,
        timeout_seconds: float = 180,
        download_limit_bytes: int = 30 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
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
        payload = {
            "model": self.model_name,
            "prompt": self._build_prompt(plan, user_instruction),
            "image": self._encode_image(source_path),
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
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
            raise SeedreamProviderError("Unsupported input image type.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        instructions = [
            plan.edit_prompt,
            "保持原图主体身份、结构、姿态与真实空间关系。",
            "自然修饰，不要引入多余人物、物体、文字、水印或明显生成痕迹。",
        ]
        if user_instruction.strip():
            instructions.append(f"用户补充要求：{user_instruction.strip()}")
        return "\n".join(instructions)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SeedreamProviderError(
                f"Seedream API returned invalid JSON (HTTP {response.status_code})."
            ) from exc

        if response.is_error or payload.get("error"):
            error = payload.get("error") or {}
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            ) or payload.get("message") or "Request failed."
            raise SeedreamProviderError(
                f"Seedream API call failed (HTTP {response.status_code}): {message}"
            )
        return payload

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        try:
            image_url = payload["data"][0]["url"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SeedreamProviderError(
                "Seedream API response did not contain an output image."
            ) from exc
        if not image_url:
            raise SeedreamProviderError(
                "Seedream API response did not contain an output image."
            )
        return str(image_url)

    def _save_download(self, response: httpx.Response, output_path: Path) -> None:
        if response.is_error:
            raise SeedreamProviderError(
                f"Failed to download Seedream output image (HTTP {response.status_code})."
            )
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise SeedreamProviderError("Seedream output URL did not return an image.")
        if len(response.content) > self.download_limit_bytes:
            raise SeedreamProviderError("Seedream output image exceeded the download limit.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
