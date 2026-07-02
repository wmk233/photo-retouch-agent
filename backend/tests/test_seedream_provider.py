import json
from pathlib import Path

import httpx

from app.core.config import Settings
from app.providers.factory import ImageProviderFactory
from app.providers.seedream_image_provider import SeedreamImageProvider
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


def test_seedream_provider_sends_image_and_downloads_result(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    source.write_bytes(make_image_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ark.example.test":
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"data": [{"url": "https://images.example.test/result.png"}]},
            )
        return httpx.Response(
            200,
            content=make_image_bytes(),
            headers={"content-type": "image/png"},
        )

    provider = SeedreamImageProvider(
        api_key="ark-test-secret",
        endpoint="https://ark.example.test/images/generations",
        model_name="doubao-seedream-5-0-260128",
        transport=httpx.MockTransport(handler),
    )

    provider.edit_image(source, output, _plan(), "背景干净一点")

    assert captured["authorization"] == "Bearer ark-test-secret"
    assert captured["payload"]["image"].startswith("data:image/png;base64,")
    assert "背景干净一点" in captured["payload"]["prompt"]
    assert captured["payload"]["model"] == "doubao-seedream-5-0-260128"
    assert output.read_bytes() == make_image_bytes()


def test_factory_creates_seedream_action(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        ark_api_key="ark-server-key",
    )

    provider = ImageProviderFactory(config).create("seedream")

    assert isinstance(provider, SeedreamImageProvider)
    assert provider.model_name == "doubao-seedream-5-0-260128"
