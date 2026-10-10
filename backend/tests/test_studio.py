"""The studio workstation: quote first, spend only on an explicit confirmation.

Every test here stays offline. ``/studio/quote`` and ``/studio/status`` are pure
catalogue math, and the two ``/studio/render`` cases are refused *before* any
request is built -- one for the missing ``confirmPaid``, one for the missing API
key -- so a green suite can never bill anything.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_storage_service,
    get_studio_interrogate_service,
    get_studio_service,
)
from app.core.config import Settings
from app.main import create_app
from app.services.storage import StorageService
from app.services.studio_interrogate import StudioInterrogateService
from app.services.studio_service import StudioService
from tests.conftest import make_portrait_bytes

INTERROGATE_KEY = "sk-studio-interrogate-secret"
INTERROGATE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


@pytest.fixture()
def studio_client(temp_settings: Settings) -> Iterator[TestClient]:
    """A client whose studio service is pinned to the offline test settings.

    The studio service is overridden rather than left on the container: the
    container reads the real ``.env``, and a render test must never be able to
    reach a live key.
    """
    app = create_app(_settings=temp_settings)
    storage = StorageService(temp_settings)
    app.dependency_overrides[get_storage_service] = lambda: storage
    app.dependency_overrides[get_studio_service] = lambda: StudioService(
        temp_settings, storage=storage
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_status_lists_the_catalogue_without_a_key(studio_client: TestClient) -> None:
    response = studio_client.get("/api/studio/status")

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert body["provider"] == "siliconflow"
    assert body["defaultEditModel"] == "Qwen/Qwen-Image-Edit-2509"
    assert body["defaultTextModel"] == "Qwen/Qwen-Image"
    assert body["maxRenders"] == 4
    assert {model["mode"] for model in body["models"]} == {"edit", "text2image"}
    edit = next(m for m in body["models"] if m["id"] == "Qwen/Qwen-Image-Edit-2509")
    assert edit["requiresSource"] is True


def test_quote_counts_one_paid_request_per_render(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/quote",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "count": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Qwen/Qwen-Image"
    assert body["mode"] == "text2image"
    assert body["count"] == 2
    assert body["paidRequests"] == 2
    assert body["imageSize"] == "1328x1328"
    assert body["requiresConfirmation"] is True
    assert "2 次真实付费请求" in body["note"]


def test_quote_clamps_a_missing_count_to_one(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/quote",
        json={"prompt": "夜景霓虹街道路口", "mode": "text2image", "count": 0},
    )

    assert response.status_code == 200
    assert response.json()["paidRequests"] == 1


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"prompt": "   ", "mode": "text2image"}, "提示词不能为空"),
        ({"prompt": "换一件大衣", "mode": "edit"}, "请先上传一张输入图"),
        ({"prompt": "出四张", "mode": "text2image", "count": 5}, "一次最多生成 4 张"),
        (
            {
                "prompt": "出图",
                "mode": "text2image",
                "modelId": "Tongyi-MAI/Z-Image",
                "imageSize": "1328x1328",
            },
            "不支持这个尺寸",
        ),
        (
            {"prompt": "出图", "mode": "text2image", "modelId": "no/such-model"},
            "还没有接入模型",
        ),
        (
            {"prompt": "出图", "mode": "text2image", "modelId": "Qwen/Qwen-Image-Edit-2509"},
            "不属于",
        ),
    ],
)
def test_quote_refuses_invalid_requests(
    studio_client: TestClient, payload: dict, detail: str
) -> None:
    response = studio_client.post("/api/studio/quote", json=payload)

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_render_without_confirmation_is_refused(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/render",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "confirmPaid": False},
    )

    assert response.status_code == 400
    assert "未确认费用" in response.json()["detail"]


def test_render_without_a_key_reports_the_missing_setting(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/render",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "confirmPaid": True},
    )

    assert response.status_code == 400
    assert "SILICONFLOW_API_KEY" in response.json()["detail"]


def test_history_is_empty_before_the_first_render(studio_client: TestClient) -> None:
    response = studio_client.get("/api/studio/history")

    assert response.status_code == 200
    assert response.json() == []


# ------------------------------------------------------- 反推提示词（interrogate）
#
# Reverse-prompting is its own paid step: quoting it never goes online, and the
# paid call is served by ``httpx.MockTransport`` so nothing billable can happen
# here. The one live call is made once, by hand, outside the suite.


@contextlib.contextmanager
def _interrogate_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> Iterator[TestClient]:
    """A studio client whose interrogator runs on the offline test settings."""
    app = create_app(_settings=settings)
    storage = StorageService(settings)
    app.dependency_overrides[get_storage_service] = lambda: storage
    app.dependency_overrides[get_studio_service] = lambda: StudioService(
        settings, storage=storage
    )
    app.dependency_overrides[get_studio_interrogate_service] = (
        lambda: StudioInterrogateService(settings, storage=storage, transport=transport)
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _upload(client: TestClient) -> str:
    response = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()["imageId"]


def _settings_with_key(tmp_path: Path) -> Settings:
    """A settings clone carrying a fake key so the interrogator builds a brain.

    The brain is always built with a ``MockTransport`` in these tests, so the
    fake key never reaches a real endpoint.
    """
    return Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        brain_provider="local",
        dashscope_api_key=None,
        dashscope_workspace_id=None,
        dashscope_endpoint=None,
        deepseek_api_key=None,
        zhipu_api_key=None,
        ark_api_key=None,
        siliconflow_api_key=INTERROGATE_KEY,
        face_lock_mode="off",
        rate_limit_per_second=0,
        max_concurrent_jobs=10,
    )


def test_status_lists_the_vision_catalogue(temp_settings: Settings) -> None:
    with _interrogate_client(temp_settings) as client:
        body = client.get("/api/studio/status").json()

    ids = [model["id"] for model in body["interrogateModels"]]
    assert INTERROGATE_MODEL in ids
    assert len(ids) == 3
    # Unconfigured means "no default to point at", not "no catalogue to browse".
    assert body["interrogateModel"] == ""


def test_interrogate_quote_prices_one_paid_request(
    temp_settings: Settings, tmp_path: Path
) -> None:
    with _interrogate_client(temp_settings) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate/quote",
            json={"sourceImageId": image_id, "focus": "style"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == INTERROGATE_MODEL
    assert body["focus"] == "style"
    assert body["sourceImageId"] == image_id
    assert body["paidRequests"] == 1
    assert body["requiresConfirmation"] is True
    assert "1 次真实付费请求" in body["note"]


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"sourceImageId": "  "}, "请先上传一张输入图"),
        ({"sourceImageId": "img_missing0000"}, "找不到这张输入图"),
    ],
)
def test_interrogate_quote_refuses_invalid_requests(
    temp_settings: Settings, payload: dict, detail: str
) -> None:
    with _interrogate_client(temp_settings) as client:
        response = client.post("/api/studio/interrogate/quote", json=payload)

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_interrogate_quote_rejects_an_unlisted_model(temp_settings: Settings) -> None:
    with _interrogate_client(temp_settings) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate/quote",
            json={"sourceImageId": image_id, "modelId": "no/such-vl"},
        )

    assert response.status_code == 400
    assert "还没有接入" in response.json()["detail"]


def test_interrogate_without_confirmation_is_refused(temp_settings: Settings) -> None:
    with _interrogate_client(temp_settings) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate",
            json={"sourceImageId": image_id, "confirmPaid": False},
        )

    assert response.status_code == 400
    assert "未确认费用" in response.json()["detail"]


def test_interrogate_without_a_key_reports_the_missing_setting(
    temp_settings: Settings,
) -> None:
    with _interrogate_client(temp_settings) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate",
            json={"sourceImageId": image_id, "confirmPaid": True},
        )

    assert response.status_code == 400
    assert "SILICONFLOW_API_KEY" in response.json()["detail"]


def test_interrogate_spends_exactly_one_paid_vision_call(tmp_path: Path) -> None:
    settings = _settings_with_key(tmp_path)
    captured: list[dict] = []
    content = json.dumps(
        {
            "prompt": "傍晚的海边礁石，人物侧身站在潮线旁，暖色逆光，浅景深",
            "negativePrompt": "避免过度磨皮",
            "alternatives": ["另一种说法", "第三种说法", "傍晚的海边礁石，人物侧身站在潮线旁，暖色逆光，浅景深"],
        },
        ensure_ascii=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    with _interrogate_client(settings, transport=httpx.MockTransport(handler)) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate",
            json={"sourceImageId": image_id, "modelId": INTERROGATE_MODEL, "confirmPaid": True},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt"].startswith("傍晚的海边礁石")
    assert body["negativePrompt"] == "避免过度磨皮"
    # The duplicate of the main prompt is dropped; three alternatives cap at two.
    assert body["alternatives"] == ["另一种说法", "第三种说法"]
    assert body["paidRequests"] == 1
    assert body["model"] == INTERROGATE_MODEL
    assert body["quote"]["paidRequests"] == 1

    # Exactly one billed request left the process, and it carried the picture.
    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == INTERROGATE_MODEL
    user_content = payload["messages"][1]["content"]
    image_part = next(part for part in user_content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_interrogate_failure_is_reported_and_the_key_is_redacted(tmp_path: Path) -> None:
    settings = _settings_with_key(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"error": {"message": f"upstream boom {INTERROGATE_KEY}"}}
        )

    with _interrogate_client(settings, transport=httpx.MockTransport(handler)) as client:
        image_id = _upload(client)
        response = client.post(
            "/api/studio/interrogate",
            json={"sourceImageId": image_id, "confirmPaid": True},
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "反推失败" in detail
    assert INTERROGATE_KEY not in detail
