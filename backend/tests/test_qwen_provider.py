import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.providers.factory import ImageProviderFactory
from app.providers.qwen_image_provider import QwenImageProvider
from app.schemas.retouch import RetouchPlan
from tests.conftest import make_image_bytes


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


def test_qwen_provider_sends_base64_and_downloads_result(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    source.write_bytes(make_image_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.example.test":
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://images.example.test/result.png"}
                                    ]
                                }
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            content=make_image_bytes(),
            headers={"content-type": "image/png"},
        )

    provider = QwenImageProvider(
        api_key="sk-test-secret",
        endpoint="https://api.example.test/generation",
        model_name="qwen-image-2.0-pro",
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.edit_image(source, output, _plan(), "肤色再亮一点"))

    content = captured["payload"]["input"]["messages"][0]["content"]
    assert captured["authorization"] == "Bearer sk-test-secret"
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert "肤色再亮一点" in content[1]["text"]
    assert captured["payload"]["model"] == "qwen-image-2.0-pro"
    assert output.read_bytes() == make_image_bytes()


def test_factory_uses_request_key_for_auto_provider(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        dashscope_api_key=None,
        dashscope_workspace_id=None,
        dashscope_endpoint="https://api.example.test/generation",
    )

    provider = ImageProviderFactory(config).create("auto", api_key="sk-user-key")

    assert isinstance(provider, QwenImageProvider)
    assert provider.model_name == "qwen-image-2.0-pro"


def test_factory_rejects_qwen_without_api_key(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        dashscope_api_key=None,
        dashscope_workspace_id=None,
        dashscope_endpoint=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        ImageProviderFactory(config).create("qwen")

    assert exc_info.value.status_code == 400
    assert "API key is required" in exc_info.value.detail


def test_factory_creates_wan_action_with_wan_parameter_profile(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        dashscope_api_key="sk-server-key",
        dashscope_workspace_id=None,
        dashscope_endpoint="https://api.example.test/generation",
    )

    provider = ImageProviderFactory(config).create("wan")

    assert isinstance(provider, QwenImageProvider)
    assert provider.provider_name == "wan"
    assert provider.model_name == "wan2.7-image-pro"
    assert provider.parameter_profile == "wan"


def test_factory_rejects_invalid_workspace_id(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        image_provider="qwen",
        dashscope_api_key="sk-server-key",
        dashscope_workspace_id=None,
        dashscope_endpoint=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        ImageProviderFactory(config).create("qwen", workspace_id="bad.example/path")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid DashScope workspace ID."
