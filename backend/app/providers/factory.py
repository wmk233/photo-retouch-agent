import re
from typing import Any

from app.core.config import Settings, settings
from app.core.errors import bad_request
from app.providers.base import ImageEditProvider
from app.providers.mock_image_provider import MockImageProvider
from app.providers.qwen_image_provider import QwenImageProvider

_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class ImageProviderFactory:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> ImageEditProvider:
        has_request_key = bool((api_key or "").strip())
        resolved_key = (api_key or self.config.dashscope_api_key or "").strip()
        name = self._resolve_provider_name(
            provider_name,
            has_request_key=has_request_key,
            has_resolved_key=bool(resolved_key),
        )

        if name in {"mock", "local"}:
            return MockImageProvider()
        if name != "qwen":
            raise bad_request(f"Unsupported image provider: {name}")
        if not resolved_key:
            raise bad_request(
                "Qwen API key is required. Set DASHSCOPE_API_KEY or provide X-AI-API-Key."
            )

        endpoint = self._resolve_endpoint(workspace_id)
        return QwenImageProvider(
            api_key=resolved_key,
            endpoint=endpoint,
            model_name=self.config.dashscope_image_model,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "defaultProvider": self.config.image_provider,
            "qwenConfigured": bool(self.config.dashscope_api_key),
            "qwenModel": self.config.dashscope_image_model,
            "workspaceConfigured": bool(
                self.config.dashscope_workspace_id or self.config.dashscope_endpoint
            ),
        }

    def _resolve_provider_name(
        self,
        requested: str | None,
        has_request_key: bool,
        has_resolved_key: bool,
    ) -> str:
        name = (requested or "auto").strip().lower()
        if name != "auto":
            return name
        if has_request_key:
            return "qwen"

        configured = self.config.image_provider
        if configured not in {"", "auto"}:
            return configured
        return "qwen" if has_resolved_key else "mock"

    def _resolve_endpoint(self, request_workspace_id: str | None) -> str:
        if self.config.dashscope_endpoint:
            return self.config.dashscope_endpoint

        workspace_id = (
            request_workspace_id or self.config.dashscope_workspace_id or ""
        ).strip()
        if workspace_id:
            if not _WORKSPACE_ID.fullmatch(workspace_id):
                raise bad_request("Invalid DashScope workspace ID.")
            region_domain = (
                "ap-southeast-1.maas.aliyuncs.com"
                if self.config.dashscope_region == "singapore"
                else "cn-beijing.maas.aliyuncs.com"
            )
            return (
                f"https://{workspace_id}.{region_domain}"
                "/api/v1/services/aigc/multimodal-generation/generation"
            )

        return (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
            "multimodal-generation/generation"
        )


provider_factory = ImageProviderFactory()
