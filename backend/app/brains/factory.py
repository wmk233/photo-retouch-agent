from typing import Any

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.brains.openai_compatible import OpenAICompatiblePromptBrain
from app.core.config import Settings, settings
from app.core.errors import bad_request


class AgentBrainFactory:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
    ) -> PromptBrain:
        name = (provider_name or self.config.brain_provider or "local").strip().lower()
        if name == "auto":
            name = self.config.brain_provider or "local"
        if name in {"local", "none"}:
            return LocalPromptBrain()

        if name == "deepseek":
            resolved_key = (api_key or self.config.deepseek_api_key or "").strip()
            if not resolved_key:
                raise bad_request(
                    "DeepSeek API key is required. Set DEEPSEEK_API_KEY or provide X-Agent-API-Key."
                )
            return OpenAICompatiblePromptBrain(
                provider_name="deepseek",
                api_key=resolved_key,
                endpoint=self.config.deepseek_endpoint,
                model_name=self.config.deepseek_model,
                timeout_seconds=self.config.provider_timeout_seconds,
            )

        if name == "glm":
            resolved_key = (api_key or self.config.zhipu_api_key or "").strip()
            if not resolved_key:
                raise bad_request(
                    "GLM API key is required. Set ZHIPU_API_KEY or provide X-Agent-API-Key."
                )
            return OpenAICompatiblePromptBrain(
                provider_name="glm",
                api_key=resolved_key,
                endpoint=self.config.zhipu_endpoint,
                model_name=self.config.zhipu_model,
                timeout_seconds=self.config.provider_timeout_seconds,
            )

        raise bad_request(f"Unsupported agent brain provider: {name}")

    def capabilities(self) -> dict[str, Any]:
        return {
            "defaultBrainProvider": self.config.brain_provider,
            "deepseekConfigured": bool(self.config.deepseek_api_key),
            "deepseekModel": self.config.deepseek_model,
            "glmConfigured": bool(self.config.zhipu_api_key),
            "glmModel": self.config.zhipu_model,
        }


agent_brain_factory = AgentBrainFactory()
