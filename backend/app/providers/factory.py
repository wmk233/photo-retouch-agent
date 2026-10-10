from __future__ import annotations
import re
from typing import Any

from app.core.config import Settings, settings
from app.core.errors import bad_request
from app.providers.base import ImageEditProvider
from app.providers.fal_inpainting_provider import FalInpaintingProvider
from app.providers.mock_image_provider import MockImageProvider
from app.providers.qwen_image_provider import QwenImageProvider
from app.providers.registry import ImageProviderPlugin, ImageProviderRegistry
from app.providers.runninghub_crop_stitch_provider import (
    RunningHubCropStitchProvider,
)
from app.providers.seedream_image_provider import SeedreamImageProvider
from app.providers.siliconflow_image_provider import SiliconFlowImageProvider

_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class ImageProviderFactory:
    def __init__(
        self,
        config: Settings = settings,
        registry: ImageProviderRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or ImageProviderRegistry()
        if registry is None:
            self._register_builtin_plugins()

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> ImageEditProvider:
        name = self._resolve_provider_name(provider_name, api_key)
        try:
            plugin = self.registry.get(name)
        except KeyError:
            raise bad_request(f"Unsupported image provider: {name}")
        return plugin.build(api_key, workspace_id)

    def register(self, plugin: ImageProviderPlugin) -> None:
        self.registry.register(plugin)

    def capabilities(self) -> dict[str, Any]:
        return {
            "defaultProvider": self.config.image_provider,
            "actionProviders": self.registry.capabilities(),
            "qwenConfigured": bool(self.config.dashscope_api_key),
            "qwenModel": self.config.dashscope_image_model,
            "wanConfigured": bool(self.config.dashscope_api_key),
            "wanModel": self.config.dashscope_wan_model,
            "seedreamConfigured": bool(self.config.ark_api_key),
            "seedreamModel": self.config.ark_image_model,
            "siliconflowConfigured": bool(self.config.siliconflow_api_key),
            "siliconflowImageModel": self.config.siliconflow_image_model,
            "falInpaintConfigured": bool(self.config.fal_key),
            "falInpaintModel": self.config.fal_inpaint_model,
            "runninghubConfigured": bool(self.config.runninghub_api_key),
            "runninghubModel": self.config.runninghub_checkpoint,
            "runninghubProviderId": "runninghub-crop-stitch",
            "workspaceConfigured": bool(
                self.config.dashscope_workspace_id or self.config.dashscope_endpoint
            ),
        }

    def _register_builtin_plugins(self) -> None:
        self.register(
            ImageProviderPlugin(
                plugin_id="qwen",
                build=lambda api_key, workspace_id: self._create_qwen_provider(
                    "qwen",
                    api_key,
                    workspace_id,
                ),
                capability=lambda: self._capability(
                    "qwen",
                    "Qwen Image",
                    self.config.dashscope_image_model,
                    bool(self.config.dashscope_api_key),
                    workspace_supported=True,
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="wan",
                build=lambda api_key, workspace_id: self._create_qwen_provider(
                    "wan",
                    api_key,
                    workspace_id,
                ),
                capability=lambda: self._capability(
                    "wan",
                    "Wan Image",
                    self.config.dashscope_wan_model,
                    bool(self.config.dashscope_api_key),
                    workspace_supported=True,
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="seedream",
                build=self._create_seedream_provider,
                capability=lambda: self._capability(
                    "seedream",
                    "Seedream",
                    self.config.ark_image_model,
                    bool(self.config.ark_api_key),
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="siliconflow",
                build=self._create_siliconflow_provider,
                capability=lambda: self._capability(
                    "siliconflow",
                    "SiliconFlow Image",
                    self.config.siliconflow_image_model,
                    bool(self.config.siliconflow_api_key),
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="fal-inpaint",
                aliases=("fal", "qwen-inpaint"),
                build=self._create_fal_inpainting_provider,
                capability=lambda: self._capability(
                    "fal-inpaint",
                    "fal.ai Qwen Inpaint",
                    self.config.fal_inpaint_model,
                    bool(self.config.fal_key),
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="mock",
                aliases=("local",),
                build=lambda _api_key, _workspace_id: MockImageProvider(),
                capability=lambda: self._capability(
                    "mock",
                    "本地模拟",
                    MockImageProvider.model_name,
                    True,
                    requires_api_key=False,
                ),
            )
        )
        self.register(
            ImageProviderPlugin(
                plugin_id="runninghub-crop-stitch",
                aliases=("runninghub",),
                build=self._create_runninghub_provider,
                capability=lambda: self._capability(
                    "runninghub-crop-stitch",
                    "RunningHub CropAndStitch",
                    self.config.runninghub_checkpoint,
                    bool(self.config.runninghub_api_key),
                ),
            )
        )

    def _create_runninghub_provider(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> ImageEditProvider:
        resolved_key = self._require_key(
            api_key,
            self.config.runninghub_api_key,
            "RunningHub",
            "RUNNINGHUB_API_KEY",
        )
        return RunningHubCropStitchProvider(
            api_key=resolved_key,
            base_url=self.config.runninghub_base_url,
            workflow_id=self.config.runninghub_workflow_id,
            instance_type=self.config.runninghub_instance_type,
            seed=self.config.runninghub_seed,
            timeout_seconds=self.config.runninghub_timeout_seconds,
            poll_interval_seconds=self.config.runninghub_poll_interval_seconds,
            checkpoint=self.config.runninghub_checkpoint,
            object_info_path=self.config.runninghub_object_info_path,
        )

    def _create_qwen_provider(
        self,
        provider_name: str,
        api_key: str | None,
        workspace_id: str | None,
    ) -> ImageEditProvider:
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
                if provider_name == "wan"
                else self.config.dashscope_image_model
            ),
            provider_name=provider_name,
            parameter_profile=provider_name,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
        )

    def _create_seedream_provider(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> ImageEditProvider:
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

    def _create_siliconflow_provider(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> ImageEditProvider:
        resolved_key = self._require_key(
            api_key,
            self.config.siliconflow_api_key,
            "SiliconFlow",
            "SILICONFLOW_API_KEY",
        )
        return SiliconFlowImageProvider(
            api_key=resolved_key,
            base_url=self.config.siliconflow_base_url,
            model_name=self.config.siliconflow_image_model,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
        )

    def _create_fal_inpainting_provider(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> ImageEditProvider:
        resolved_key = self._require_key(
            api_key,
            self.config.fal_key,
            "fal.ai",
            "FAL_KEY",
        )
        return FalInpaintingProvider(
            api_key=resolved_key,
            endpoint=self.config.fal_inpaint_endpoint,
            model_name=self.config.fal_inpaint_model,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
            poll_interval_seconds=self.config.fal_inpaint_poll_interval_seconds,
            max_poll_seconds=self.config.fal_inpaint_max_poll_seconds,
        )

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
