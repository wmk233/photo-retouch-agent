from __future__ import annotations
import re
from typing import Any

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.brains.openai_compatible import OpenAICompatiblePromptBrain
from app.brains.registry import AgentBrainPlugin, AgentBrainRegistry
from app.core.config import Settings, settings
from app.core.errors import bad_request

_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class AgentBrainFactory:
    def __init__(
        self,
        config: Settings = settings,
        registry: AgentBrainRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or AgentBrainRegistry()
        if registry is None:
            self._register_builtin_plugins()

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> PromptBrain:
        name = (provider_name or self.config.brain_provider or "local").strip().lower()
        try:
            plugin = self.registry.get(name)
        except KeyError:
            raise bad_request(f"Unsupported agent brain provider: {name}")
        return plugin.build(api_key, workspace_id)

    def register(self, plugin: AgentBrainPlugin) -> None:
        self.registry.register(plugin)

    def capabilities(self) -> dict[str, Any]:
        return {
            "defaultBrainProvider": self.config.brain_provider,
            "brainProviders": self.registry.capabilities(),
            "qwenVisionConfigured": bool(self.config.dashscope_api_key),
            "qwenVisionModel": self.config.dashscope_vision_model,
            "deepseekConfigured": bool(self.config.deepseek_api_key),
            "deepseekModel": self.config.deepseek_model,
            "glmConfigured": bool(self.config.zhipu_api_key),
            "glmModel": self.config.zhipu_model,
            "doubaoConfigured": bool(self.config.ark_api_key),
            "doubaoModel": self.config.ark_vision_model,
            "siliconflowVisionConfigured": bool(self.config.siliconflow_api_key),
            "siliconflowVisionModel": self.config.siliconflow_vision_model,
        }

    def _register_builtin_plugins(self) -> None:
        self.register(
            AgentBrainPlugin(
                plugin_id="qwen",
                build=self._create_qwen_brain,
                capability=lambda: self._capability(
                    "qwen",
                    "Qwen Vision",
                    self.config.dashscope_vision_model,
                    bool(self.config.dashscope_api_key),
                    "direct",
                    workspace_supported=True,
                ),
            )
        )
        self.register(
            AgentBrainPlugin(
                plugin_id="glm",
                build=self._create_glm_brain,
                capability=lambda: self._capability(
                    "glm",
                    "GLM Vision",
                    self.config.zhipu_model,
                    bool(self.config.zhipu_api_key),
                    "direct",
                ),
            )
        )
        self.register(
            AgentBrainPlugin(
                plugin_id="doubao",
                build=self._create_doubao_brain,
                capability=lambda: self._capability(
                    "doubao",
                    "豆包 Vision",
                    self.config.ark_vision_model,
                    bool(self.config.ark_api_key),
                    "direct",
                ),
            )
        )
        self.register(
            AgentBrainPlugin(
                plugin_id="siliconflow",
                build=self._create_siliconflow_brain,
                capability=lambda: self._capability(
                    "siliconflow",
                    "SiliconFlow Vision",
                    self.config.siliconflow_vision_model,
                    bool(self.config.siliconflow_api_key),
                    "direct",
                ),
            )
        )
        self.register(
            AgentBrainPlugin(
                plugin_id="deepseek",
                build=self._create_deepseek_brain,
                capability=lambda: self._capability(
                    "deepseek",
                    "DeepSeek",
                    self.config.deepseek_model,
                    bool(self.config.deepseek_api_key),
                    "derived",
                ),
            )
        )
        self.register(
            AgentBrainPlugin(
                plugin_id="local",
                aliases=("none",),
                build=lambda _api_key, _workspace_id: LocalPromptBrain(),
                capability=lambda: self._capability(
                    "local",
                    "本地规则",
                    LocalPromptBrain.model_name,
                    True,
                    "derived",
                    requires_api_key=False,
                ),
            )
        )

    def _create_deepseek_brain(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> PromptBrain:
        return OpenAICompatiblePromptBrain(
            provider_name="deepseek",
            api_key=self._require_key(
                api_key,
                self.config.deepseek_api_key,
                "DeepSeek",
                "DEEPSEEK_API_KEY",
            ),
            endpoint=self.config.deepseek_endpoint,
            model_name=self.config.deepseek_model,
            vision_mode="derived",
            timeout_seconds=self.config.provider_timeout_seconds,
        )

    def _create_glm_brain(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> PromptBrain:
        return OpenAICompatiblePromptBrain(
            provider_name="glm",
            api_key=self._require_key(
                api_key,
                self.config.zhipu_api_key,
                "GLM",
                "ZHIPU_API_KEY",
            ),
            endpoint=self.config.zhipu_endpoint,
            model_name=self.config.zhipu_model,
            vision_mode="direct",
            image_url_mode="raw_base64",
            timeout_seconds=self.config.provider_timeout_seconds,
        )

    def _create_qwen_brain(
        self,
        api_key: str | None,
        workspace_id: str | None,
    ) -> PromptBrain:
        return OpenAICompatiblePromptBrain(
            provider_name="qwen",
            api_key=self._require_key(
                api_key,
                self.config.dashscope_api_key,
                "Qwen",
                "DASHSCOPE_API_KEY",
            ),
            endpoint=self._resolve_qwen_endpoint(workspace_id),
            model_name=self.config.dashscope_vision_model,
            vision_mode="direct",
            timeout_seconds=self.config.provider_timeout_seconds,
        )

    def _create_doubao_brain(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> PromptBrain:
        return OpenAICompatiblePromptBrain(
            provider_name="doubao",
            api_key=self._require_key(
                api_key,
                self.config.ark_api_key,
                "Doubao",
                "ARK_API_KEY",
            ),
            endpoint=self.config.ark_chat_endpoint,
            model_name=self.config.ark_vision_model,
            vision_mode="direct",
            timeout_seconds=self.config.provider_timeout_seconds,
        )

    def _create_siliconflow_brain(
        self,
        api_key: str | None,
        _workspace_id: str | None,
    ) -> PromptBrain:
        return OpenAICompatiblePromptBrain(
            provider_name="siliconflow",
            api_key=self._require_key(
                api_key,
                self.config.siliconflow_api_key,
                "SiliconFlow",
                "SILICONFLOW_API_KEY",
            ),
            endpoint=f"{self.config.siliconflow_base_url}/chat/completions",
            model_name=self.config.siliconflow_vision_model,
            vision_mode="direct",
            timeout_seconds=self.config.provider_timeout_seconds,
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
                f"Set {env_name} or provide X-Agent-API-Key."
            )
        return resolved_key

    @staticmethod
    def _capability(
        provider_id: str,
        label: str,
        model: str,
        configured: bool,
        vision_mode: str,
        requires_api_key: bool = True,
        workspace_supported: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": provider_id,
            "label": label,
            "model": model,
            "configured": configured,
            "visionMode": vision_mode,
            "requiresApiKey": requires_api_key,
            "workspaceSupported": workspace_supported,
        }

    def _resolve_qwen_endpoint(self, request_workspace_id: str | None) -> str:
        if self.config.dashscope_vision_endpoint:
            return self.config.dashscope_vision_endpoint

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
                "/compatible-mode/v1/chat/completions"
            )

        legacy_domain = (
            "dashscope-intl.aliyuncs.com"
            if self.config.dashscope_region == "singapore"
            else "dashscope.aliyuncs.com"
        )
        return f"https://{legacy_domain}/compatible-mode/v1/chat/completions"
