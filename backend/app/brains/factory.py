import re
from typing import Any

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.brains.openai_compatible import OpenAICompatiblePromptBrain
from app.core.config import Settings, settings
from app.core.errors import bad_request

_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class AgentBrainFactory:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> PromptBrain:
        name = (provider_name or self.config.brain_provider or "local").strip().lower()
        if name in {"local", "none"}:
            return LocalPromptBrain()

        if name == "deepseek":
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

        if name == "glm":
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

        if name == "qwen":
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

        if name == "doubao":
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

        raise bad_request(f"Unsupported agent brain provider: {name}")

    def capabilities(self) -> dict[str, Any]:
        brain_providers = [
            self._capability(
                "qwen",
                "Qwen Vision",
                self.config.dashscope_vision_model,
                bool(self.config.dashscope_api_key),
                "direct",
                workspace_supported=True,
            ),
            self._capability(
                "glm",
                "GLM Vision",
                self.config.zhipu_model,
                bool(self.config.zhipu_api_key),
                "direct",
            ),
            self._capability(
                "doubao",
                "豆包 Vision",
                self.config.ark_vision_model,
                bool(self.config.ark_api_key),
                "direct",
            ),
            self._capability(
                "deepseek",
                "DeepSeek",
                self.config.deepseek_model,
                bool(self.config.deepseek_api_key),
                "derived",
            ),
            self._capability(
                "local",
                "本地规则",
                LocalPromptBrain.model_name,
                True,
                "derived",
                requires_api_key=False,
            ),
        ]
        return {
            "defaultBrainProvider": self.config.brain_provider,
            "brainProviders": brain_providers,
            "qwenVisionConfigured": bool(self.config.dashscope_api_key),
            "qwenVisionModel": self.config.dashscope_vision_model,
            "deepseekConfigured": bool(self.config.deepseek_api_key),
            "deepseekModel": self.config.deepseek_model,
            "glmConfigured": bool(self.config.zhipu_api_key),
            "glmModel": self.config.zhipu_model,
            "doubaoConfigured": bool(self.config.ark_api_key),
            "doubaoModel": self.config.ark_vision_model,
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


agent_brain_factory = AgentBrainFactory()
