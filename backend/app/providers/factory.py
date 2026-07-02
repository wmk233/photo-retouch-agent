import re
from typing import Any

from app.core.config import Settings, settings
from app.core.errors import bad_request
from app.providers.base import ImageEditProvider
from app.providers.mock_image_provider import MockImageProvider
from app.providers.qwen_image_provider import QwenImageProvider
from app.providers.seedream_image_provider import SeedreamImageProvider

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
        name = self._resolve_provider_name(provider_name, api_key)
        if name in {"mock", "local"}:
            return MockImageProvider()
        if name == "seedream":
            resolved_key = self._require_key(
                api_key,
                self.config.ark_api_key,
                "Seedream",
                "ARK_API_KEY",
            )
            return SeedreamImageProvider(
                api_key=resolved_key,
                endpoint=self.config.ark_image_endpoint,
                model_name=self.config.ark_image_model,
                timeout_seconds=self.config.provider_timeout_seconds,
                download_limit_bytes=self.config.provider_download_limit_bytes,
            )
        if name not in {"qwen", "wan"}:
            raise bad_request(f"Unsupported image provider: {name}")

        resolved_key = self._require_key(
            api_key,
            self.config.dashscope_api_key,
            "DashScope",
            "DASHSCOPE_API_KEY",
        )
        return QwenImageProvider(
            api_key=resolved_key,
            endpoint=self._resolve_endpoint(workspace_id),
            model_name=(
                self.config.dashscope_wan_model
                if name == "wan"
                else self.config.dashscope_image_model
            ),
            provider_name=name,
            parameter_profile=name,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
        )

    def capabilities(self) -> dict[str, Any]:
        action_providers = [
            self._capability(
                "qwen",
                "Qwen Image",
                self.config.dashscope_image_model,
                bool(self.config.dashscope_api_key),
                workspace_supported=True,
            ),
            self._capability(
                "wan",
                "Wan Image",
                self.config.dashscope_wan_model,
                bool(self.config.dashscope_api_key),
                workspace_supported=True,
            ),
            self._capability(
                "seedream",
                "Seedream",
                self.config.ark_image_model,
                bool(self.config.ark_api_key),
            ),
            self._capability(
                "mock",
                "本地模拟",
                MockImageProvider.model_name,
                True,
                requires_api_key=False,
            ),
        ]
        return {
            "defaultProvider": self.config.image_provider,
            "actionProviders": action_providers,
            "qwenConfigured": bool(self.config.dashscope_api_key),
            "qwenModel": self.config.dashscope_image_model,
            "wanConfigured": bool(self.config.dashscope_api_key),
            "wanModel": self.config.dashscope_wan_model,
            "seedreamConfigured": bool(self.config.ark_api_key),
            "seedreamModel": self.config.ark_image_model,
            "workspaceConfigured": bool(
                self.config.dashscope_workspace_id or self.config.dashscope_endpoint
            ),
        }

    @staticmethod
    def _require_key(
        request_key: str | None,
        configured_key: str | None,
        provider_label: str,
        env_name: str,
    ) -> str:
        resolved_key = (request_key or configured_key or "").strip()
        if not resolved_key:
            raise bad_request(
                f"{provider_label} API key is required. "
                f"Set {env_name} or provide X-Action-API-Key."
            )
        return resolved_key

    @staticmethod
    def _capability(
        provider_id: str,
        label: str,
        model: str,
        configured: bool,
        requires_api_key: bool = True,
        workspace_supported: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": provider_id,
            "label": label,
            "model": model,
            "configured": configured,
            "requiresApiKey": requires_api_key,
            "workspaceSupported": workspace_supported,
        }

    def _resolve_provider_name(
        self,
        requested: str | None,
        request_key: str | None,
    ) -> str:
        name = (requested or "auto").strip().lower()
        if name != "auto":
            return name
        if (request_key or "").strip():
            return "qwen"

        configured = self.config.image_provider
        if configured not in {"", "auto"}:
            return configured
        return "qwen" if self.config.dashscope_api_key else "mock"

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
