import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from app.brains.factory import AgentBrainFactory
from app.brains.openai_compatible import OpenAICompatiblePromptBrain
from app.core.config import Settings
from app.schemas.retouch import RetouchPlan


def _plan() -> RetouchPlan:
    return RetouchPlan(
        plan_id="natural",
        domain_type="portrait",
        title="自然美化",
        description="自然提亮",
        intensity="natural",
        edit_prompt="轻微提亮人物并统一肤色。",
        negative_prompt="避免改变五官",
        expected_changes=["提亮"],
    )


def test_openai_compatible_brain_optimizes_plan() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                '{"editPrompt":"自然提亮面部，保留本人特征",'
                                '"negativePrompt":"避免过度磨皮",'
                                '"expectedChanges":["面部提亮","肤色均匀"]}'
                                "\n```"
                            )
                        }
                    }
                ]
            },
        )

    brain = OpenAICompatiblePromptBrain(
        provider_name="deepseek",
        api_key="sk-brain-secret",
        endpoint="https://brain.example.test/chat/completions",
        model_name="deepseek-v4-flash",
        transport=httpx.MockTransport(handler),
    )

    optimized = brain.optimize(_plan(), "黑眼圈淡一点")

    assert captured["authorization"] == "Bearer sk-brain-secret"
    assert captured["payload"]["model"] == "deepseek-v4-flash"
    assert "黑眼圈淡一点" in captured["payload"]["messages"][1]["content"]
    assert optimized.edit_prompt == "自然提亮面部，保留本人特征"
    assert optimized.negative_prompt == "避免过度磨皮"
    assert optimized.expected_changes == ["面部提亮", "肤色均匀"]


@pytest.mark.parametrize(
    ("provider_name", "key_field", "expected_model"),
    [
        ("deepseek", "deepseek_api_key", "deepseek-v4-flash"),
        ("glm", "zhipu_api_key", "glm-5.1"),
    ],
)
def test_brain_factory_supports_deepseek_and_glm(
    tmp_path: Path,
    provider_name: str,
    key_field: str,
    expected_model: str,
) -> None:
    config_kwargs = {
        "data_dir": tmp_path / "data",
        "brain_provider": "local",
        "deepseek_api_key": None,
        "zhipu_api_key": None,
    }
    config_kwargs[key_field] = "sk-server-key"
    factory = AgentBrainFactory(Settings(**config_kwargs))

    brain = factory.create(provider_name)

    assert isinstance(brain, OpenAICompatiblePromptBrain)
    assert brain.provider_name == provider_name
    assert brain.model_name == expected_model


@pytest.mark.parametrize("provider_name", ["deepseek", "glm"])
def test_brain_factory_requires_key(
    tmp_path: Path,
    provider_name: str,
) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        brain_provider="local",
        deepseek_api_key=None,
        zhipu_api_key=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        AgentBrainFactory(config).create(provider_name)

    assert exc_info.value.status_code == 400
    assert "API key is required" in exc_info.value.detail
