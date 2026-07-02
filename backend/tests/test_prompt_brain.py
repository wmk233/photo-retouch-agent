import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from PIL import Image

from app.brains.factory import AgentBrainFactory
from app.brains.openai_compatible import OpenAICompatiblePromptBrain
from app.core.config import Settings
from app.schemas.analysis import PhotoAnalysis, SubjectSummary
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


def _analysis() -> PhotoAnalysis:
    return PhotoAnalysis(
        image_id="img_test",
        domain_type="portrait",
        scene_type="人像",
        subjects=SubjectSummary(count=1, position="中央", face_visibility="高"),
        lighting_issues=["偏暗"],
        background_issues=["轻微杂乱"],
        portrait_suggestions=["自然提亮"],
        composition_suggestions=["保持构图"],
        recommended_styles=["自然"],
        risk_flags=[],
    )


def _image_path(tmp_path: Path) -> Path:
    path = tmp_path / "portrait.png"
    Image.new("RGB", (32, 24), color=(36, 124, 114)).save(path)
    return path


def test_visual_brain_analyzes_image_and_optimizes_plan(tmp_path: Path) -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        user_content = payload["messages"][1]["content"]
        assert user_content[0]["type"] == "image_url"
        assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        if len(captured) == 1:
            content = json.dumps(
                {
                    "domainType": "landscape",
                    "sceneType": "旅行人像",
                    "subjects": {
                        "count": 1,
                        "position": "画面中央",
                        "faceVisibility": "高",
                    },
                    "lightingIssues": ["人物面部略暗"],
                    "backgroundIssues": ["背景有路人"],
                    "portraitSuggestions": ["轻微提亮肤色"],
                    "compositionSuggestions": ["保留环境"],
                    "recommendedStyles": ["自然旅行"],
                    "riskFlags": [],
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "editPrompt": "自然提亮面部，保留旅行环境和本人特征",
                    "negativePrompt": "避免过度磨皮",
                    "expectedChanges": ["面部提亮", "肤色均匀"],
                },
                ensure_ascii=False,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    brain = OpenAICompatiblePromptBrain(
        provider_name="qwen",
        api_key="sk-brain-secret",
        endpoint="https://brain.example.test/chat/completions",
        model_name="qwen3-vl-plus",
        vision_mode="direct",
        transport=httpx.MockTransport(handler),
    )
    image_path = _image_path(tmp_path)

    analysis = asyncio.run(brain.analyze("img_test", image_path, _analysis()))
    optimized = asyncio.run(brain.optimize(image_path, _plan(), "黑眼圈淡一点"))

    assert analysis.scene_type == "旅行人像"
    assert analysis.domain_type == "landscape"
    assert analysis.brain_provider == "qwen"
    assert analysis.brain_model == "qwen3-vl-plus"
    assert analysis.vision_mode == "direct"
    assert optimized.edit_prompt == "自然提亮面部，保留旅行环境和本人特征"
    assert optimized.negative_prompt == "避免过度磨皮"
    assert optimized.expected_changes == ["面部提亮", "肤色均匀"]


def test_visual_brain_normalizes_subject_list_from_provider(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "domainType": "portrait",
                                    "sceneType": "证件人像",
                                    "subjects": [
                                        {
                                            "type": "person",
                                            "position": "画面中央",
                                            "description": "正面人像",
                                        }
                                    ],
                                    "lightingIssues": [],
                                    "backgroundIssues": [],
                                    "portraitSuggestions": ["自然提亮"],
                                    "compositionSuggestions": ["保持构图"],
                                    "recommendedStyles": ["自然"],
                                    "riskFlags": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    brain = OpenAICompatiblePromptBrain(
        provider_name="qwen",
        api_key="sk-brain-secret",
        endpoint="https://brain.example.test/chat/completions",
        model_name="qwen3-vl-plus",
        vision_mode="direct",
        transport=httpx.MockTransport(handler),
    )

    analysis = asyncio.run(brain.analyze("img_test", _image_path(tmp_path), _analysis()))

    assert analysis.subjects.count == 1
    assert analysis.subjects.position == "画面中央"
    assert analysis.subjects.face_visibility == "高"


def test_derived_brain_does_not_claim_direct_image_input(tmp_path: Path) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "editPrompt": "根据结构化分析自然提亮",
                                    "negativePrompt": "避免失真",
                                    "expectedChanges": ["提亮"],
                                },
                                ensure_ascii=False,
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
        vision_mode="derived",
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(brain.optimize(_image_path(tmp_path), _plan(), ""))

    assert isinstance(captured["messages"][1]["content"], str)
    assert "无图片输入能力" in captured["messages"][1]["content"]


@pytest.mark.parametrize(
    ("provider_name", "key_field", "expected_model", "vision_mode"),
    [
        ("deepseek", "deepseek_api_key", "deepseek-v4-flash", "derived"),
        ("glm", "zhipu_api_key", "glm-5v-turbo", "direct"),
        ("qwen", "dashscope_api_key", "qwen3-vl-plus", "direct"),
        ("doubao", "ark_api_key", "doubao-seed-2-0-lite-260215", "direct"),
    ],
)
def test_brain_factory_supports_configured_providers(
    tmp_path: Path,
    provider_name: str,
    key_field: str,
    expected_model: str,
    vision_mode: str,
) -> None:
    config_kwargs = {
        "data_dir": tmp_path / "data",
        "brain_provider": "local",
        "deepseek_api_key": None,
        "zhipu_api_key": None,
        "dashscope_api_key": None,
        "ark_api_key": None,
    }
    config_kwargs[key_field] = "sk-server-key"
    factory = AgentBrainFactory(Settings(**config_kwargs))

    brain = factory.create(provider_name)

    assert isinstance(brain, OpenAICompatiblePromptBrain)
    assert brain.provider_name == provider_name
    assert brain.model_name == expected_model
    assert brain.vision_mode == vision_mode


@pytest.mark.parametrize("provider_name", ["deepseek", "glm", "qwen", "doubao"])
def test_brain_factory_requires_key(
    tmp_path: Path,
    provider_name: str,
) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        brain_provider="local",
        dashscope_api_key=None,
        deepseek_api_key=None,
        zhipu_api_key=None,
        ark_api_key=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        AgentBrainFactory(config).create(provider_name)

    assert exc_info.value.status_code == 400
    assert "API key is required" in exc_info.value.detail
