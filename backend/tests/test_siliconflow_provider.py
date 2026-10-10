"""SiliconFlow provider tests: vision brain plus image editing.

Every HTTP call below is served by ``httpx.MockTransport``. No request leaves the
machine and no billable API is touched.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_agent_brain_factory,
    get_job_store,
    get_provider_factory,
    get_storage_service,
)
from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.main import create_app
from app.providers.factory import ImageProviderFactory
from app.providers.mock_image_provider import MockImageProvider
from app.providers.siliconflow_image_provider import (
    SiliconFlowImageProvider,
    SiliconFlowProviderError,
)
from app.schemas.retouch import RetouchPlan
from app.services.job_store import JobStore
from app.services.storage import StorageService
from tests.conftest import make_image_bytes, make_landscape_bytes, make_portrait_bytes

VISION_MODEL = "Qwen/Qwen3.5-35B-A3B"
IMAGE_MODEL = "Qwen/Qwen-Image-Edit-2509"
API_KEY = "sk-siliconflow-test-secret"
BASE_URL = "https://api.siliconflow.cn/v1"


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "image_provider": "mock",
        "brain_provider": "local",
        "dashscope_api_key": None,
        "ark_api_key": None,
        "zhipu_api_key": None,
        "deepseek_api_key": None,
        "siliconflow_api_key": None,
        # These suites drive the provider/job plumbing with synthetic imagery that
        # has no real face; the strict identity lock is exercised separately in
        # tests/test_face_lock.py with an injected stub detector.
        "face_lock_mode": "off",
        "rate_limit_per_second": 0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _plan(plan_id: str = "natural", prompt: str = "自然提亮肤色并保留本人特征。") -> RetouchPlan:
    return RetouchPlan(
        plan_id=plan_id,
        domain_type="portrait",
        title=plan_id,
        description=plan_id,
        intensity="natural",
        edit_prompt=prompt,
        negative_prompt="避免改变五官",
        expected_changes=["提亮"],
    )


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Force every provider/brain client built in-process onto a mock transport."""
    real_client = httpx.AsyncClient

    def build(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("app.brains.openai_compatible.httpx.AsyncClient", build)


def test_siliconflow_is_registered_for_vision_and_editing(tmp_path: Path) -> None:
    without_key = _settings(tmp_path)
    action_caps = ImageProviderFactory(without_key).capabilities()
    entry = next(item for item in action_caps["actionProviders"] if item["id"] == "siliconflow")
    assert entry["model"] == IMAGE_MODEL
    assert entry["configured"] is False
    assert entry["requiresApiKey"] is True
    assert action_caps["siliconflowConfigured"] is False
    assert action_caps["siliconflowImageModel"] == IMAGE_MODEL

    brain_caps = AgentBrainFactory(without_key).capabilities()
    brain_entry = next(
        item for item in brain_caps["brainProviders"] if item["id"] == "siliconflow"
    )
    assert brain_entry["model"] == VISION_MODEL
    assert brain_entry["visionMode"] == "direct"
    assert brain_entry["configured"] is False
    assert brain_caps["siliconflowVisionConfigured"] is False

    with_key = _settings(tmp_path, siliconflow_api_key=API_KEY)
    assert ImageProviderFactory(with_key).capabilities()["siliconflowConfigured"] is True
    assert AgentBrainFactory(with_key).capabilities()["siliconflowVisionConfigured"] is True


def test_missing_key_is_rejected_before_any_request(tmp_path: Path) -> None:
    config = _settings(tmp_path)

    with pytest.raises(HTTPException) as action_error:
        ImageProviderFactory(config).create("siliconflow")
    assert action_error.value.status_code == 400
    assert "SILICONFLOW_API_KEY" in action_error.value.detail
    assert API_KEY not in action_error.value.detail

    with pytest.raises(HTTPException) as brain_error:
        AgentBrainFactory(config).create("siliconflow")
    assert brain_error.value.status_code == 400
    assert "SILICONFLOW_API_KEY" in brain_error.value.detail
    assert "X-Agent-API-Key" in brain_error.value.detail


def test_siliconflow_brain_sends_the_real_source_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "portrait.png"
    source.write_bytes(make_portrait_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "editPrompt": "提亮面部并保留本人特征",
                                    "negativePrompt": "不要磨皮过度",
                                    "expectedChanges": ["面部提亮"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    _install_mock_transport(monkeypatch, handler)
    brain = AgentBrainFactory(_settings(tmp_path, siliconflow_api_key=API_KEY)).create(
        "siliconflow"
    )

    plan = asyncio.run(brain.optimize(source, _plan(), "黑眼圈淡一点"))

    assert captured["url"] == f"{BASE_URL}/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == VISION_MODEL
    assert payload["messages"][0]["role"] == "system"
    image_part = payload["messages"][1]["content"][0]
    assert image_part["type"] == "image_url"
    data_url = image_part["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == source.read_bytes()
    assert plan.edit_prompt == "提亮面部并保留本人特征"
    assert brain.provider_name == "siliconflow"
    assert brain.vision_mode == "direct"


def test_image_edit_sends_image_and_prompt_without_unsupported_size(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out_test.png"
    source.write_bytes(make_image_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"x-request-id": "req-123"},
                json={
                    "images": [{"url": "https://images.example.test/result.png"}],
                    "timings": {"inference": 1.5},
                },
            )
        return httpx.Response(
            200,
            content=make_image_bytes(),
            headers={"content-type": "image/png"},
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.edit_image(source, output, _plan(), "背景干净一点"))

    assert captured["url"] == f"{BASE_URL}/images/generations"
    assert captured["authorization"] == f"Bearer {API_KEY}"
    payload = captured["payload"]
    assert payload["model"] == IMAGE_MODEL
    assert payload["image"].startswith("data:image/png;base64,")
    assert base64.b64decode(payload["image"].split(",", 1)[1]) == source.read_bytes()
    assert "背景干净一点" in payload["prompt"]
    assert "image_size" not in payload
    assert "size" not in payload
    assert "image2" not in payload


def test_result_is_saved_locally_with_provenance_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    outputs = tmp_path / "outputs"
    output = outputs / "out_abc.png"
    source.write_bytes(make_image_bytes())
    result_bytes = make_image_bytes(size=(16, 16))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"x-request-id": "req-abc"},
                json={"images": [{"url": "https://images.example.test/result.png"}]},
            )
        return httpx.Response(
            200, content=result_bytes, headers={"content-type": "image/png"}
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.edit_image(source, output, _plan("bold", "大胆造型。"), ""))

    assert output.read_bytes() == result_bytes
    metadata_path = outputs / "_meta" / f"{output.name}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["provider"] == "siliconflow"
    assert metadata["model"] == IMAGE_MODEL
    assert metadata["planId"] == "bold"
    assert metadata["requestId"] == "req-abc"
    assert metadata["imageCount"] == 1
    assert metadata["bytes"] == len(result_bytes)
    assert API_KEY not in metadata_path.read_text(encoding="utf-8")
    # Metadata must never be mistaken for the generated picture itself.
    assert [path.name for path in outputs.glob(f"{output.stem}.*")] == [output.name]


def test_three_directions_produce_independent_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    outputs = tmp_path / "outputs"
    source.write_bytes(make_image_bytes())
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            prompts.append(json.loads(request.content)["prompt"])
            return httpx.Response(
                200, json={"images": [{"url": "https://images.example.test/r.png"}]}
            )
        return httpx.Response(
            200, content=make_image_bytes(), headers={"content-type": "image/png"}
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )
    directions = [
        ("natural", "自然优化：轻微提亮与控油。"),
        ("feature", "特征强化：放大本人辨识度。"),
        ("bold", "大胆突破：尝试明显不同的造型方向。"),
    ]

    for plan_id, prompt in directions:
        asyncio.run(
            provider.edit_image(
                source, outputs / f"out_{plan_id}.png", _plan(plan_id, prompt), ""
            )
        )

    assert len(prompts) == 3
    assert len(set(prompts)) == 3
    for (_, prompt), sent in zip(directions, prompts):
        assert prompt in sent
    # Every direction keeps the same identity-preservation constraints.
    assert all("不换脸" in sent for sent in prompts)


def test_failure_does_not_fall_back_to_mock_or_text_to_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out_fail.png"
    source.write_bytes(make_image_bytes())
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        asyncio.run(provider.edit_image(source, output, _plan(), ""))

    assert "rate limited" in str(error.value)
    assert not output.exists()
    assert len(requests) == 1
    assert requests[0]["image"].startswith("data:image/png;base64,")


def test_provider_errors_never_leak_the_api_key(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out.png"
    source.write_bytes(make_image_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {API_KEY}"}})

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        asyncio.run(provider.edit_image(source, output, _plan(), ""))

    assert API_KEY not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_download_of_a_non_image_body_is_rejected_and_diagnosable(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out_nonimage.png"
    source.write_bytes(make_image_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"images": [{"url": "https://images.example.test/result.png"}]},
            )
        return httpx.Response(
            200,
            content=b"<html>sign in to continue</html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        asyncio.run(provider.edit_image(source, output, _plan(), ""))

    message = str(error.value)
    # Still a hard failure: no fallback, no text-to-image, no half-written file.
    assert "did not return an image" in message
    assert "text/html" in message
    assert not output.exists()


def _editing_provider(handler, **kwargs: object) -> SiliconFlowImageProvider:
    return SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


def _serving_handler(result_url: str, body: bytes, content_type: str | None):
    """A POST that hands back ``result_url`` plus a GET that serves ``body``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"images": [{"url": result_url}]})
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(200, content=body, headers=headers)

    return handler


def _run_edit(provider: SiliconFlowImageProvider, tmp_path: Path, name: str) -> Path:
    source = tmp_path / "source.png"
    source.write_bytes(make_image_bytes())
    output = tmp_path / "outputs" / name
    asyncio.run(provider.edit_image(source, output, _plan(), ""))
    return output


def test_octet_stream_response_with_real_image_bytes_is_accepted(tmp_path: Path) -> None:
    """The CDN mislabels its output; the bytes are what must decide."""
    result_bytes = make_image_bytes(size=(16, 16))
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png?Signature=abc&Expires=1",
            result_bytes,
            "application/octet-stream",
        )
    )

    output = _run_edit(provider, tmp_path, "out_octet.png")

    assert output.read_bytes() == result_bytes
    metadata_text = (output.parent / "_meta" / f"{output.name}.json").read_text("utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["contentType"] == "application/octet-stream"
    assert metadata["sourceHost"] == "images.example.test"
    # The signed query string is never persisted anywhere on disk.
    assert "Signature" not in metadata_text
    assert "http" not in metadata_text


def test_binary_content_type_without_a_content_type_header_is_accepted(
    tmp_path: Path,
) -> None:
    result_bytes = make_image_bytes(size=(16, 16))
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png", result_bytes, None
        )
    )

    output = _run_edit(provider, tmp_path, "out_noheader.png")

    assert output.read_bytes() == result_bytes


def test_octet_stream_error_page_is_still_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png",
            b"<html>sign in to continue</html>",
            "application/octet-stream",
        )
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_octet_html.png")

    message = str(error.value)
    # Rejected one layer later, on the decoded bytes rather than the declaration.
    assert "could not be decoded" in message
    assert "application/octet-stream" in message
    assert not (tmp_path / "outputs" / "out_octet_html.png").exists()


def test_image_content_type_with_invalid_bytes_is_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png", b"not really a picture", "image/png"
        )
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_fake_png.png")

    assert "could not be decoded" in str(error.value)
    assert not (tmp_path / "outputs" / "out_fake_png.png").exists()


def test_empty_body_is_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler("https://images.example.test/result.png", b"", "image/png")
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_empty.png")

    assert "was empty" in str(error.value)


def test_truncated_image_is_rejected(tmp_path: Path) -> None:
    whole = make_image_bytes(size=(64, 64))
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png", whole[: len(whole) // 2], "image/png"
        )
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_truncated.png")

    assert "could not be decoded" in str(error.value)
    assert not (tmp_path / "outputs" / "out_truncated.png").exists()


def test_unsupported_image_format_is_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png",
            make_image_bytes(image_format="BMP"),
            "image/bmp",
        )
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_bmp.png")

    assert "not supported" in str(error.value)


def test_oversized_download_is_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png", make_image_bytes(size=(64, 64)), "image/png"
        ),
        download_limit_bytes=64,
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_big.png")

    assert "exceeded the download limit" in str(error.value)


def test_redirect_to_a_non_public_address_is_rejected(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"images": [{"url": "https://images.example.test/result.png"}]}
            )
        return httpx.Response(
            302, headers={"location": "http://127.0.0.1:8000/data/outputs/other.png"}
        )

    provider = _editing_provider(handler)

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_ssrf.png")

    assert "non-public address" in str(error.value)
    assert not (tmp_path / "outputs" / "out_ssrf.png").exists()


def test_non_http_download_scheme_is_rejected(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler("file:///etc/passwd", b"root:x:0:0", "text/plain")
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_scheme.png")

    assert "not an http(s) address" in str(error.value)


def test_redirect_chain_is_followed_to_a_public_host(tmp_path: Path) -> None:
    visited: list[str] = []
    result_bytes = make_image_bytes(size=(8, 8))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"images": [{"url": "https://images.example.test/result.png"}]}
            )
        visited.append(str(request.url))
        if str(request.url).endswith("result.png"):
            return httpx.Response(
                302, headers={"location": "https://cdn.example.test/final.png"}
            )
        return httpx.Response(
            200, content=result_bytes, headers={"content-type": "application/octet-stream"}
        )

    output = _run_edit(_editing_provider(handler), tmp_path, "out_redirect.png")

    assert visited == [
        "https://images.example.test/result.png",
        "https://cdn.example.test/final.png",
    ]
    assert output.read_bytes() == result_bytes
    metadata = json.loads((output.parent / "_meta" / f"{output.name}.json").read_text("utf-8"))
    assert metadata["sourceHost"] == "cdn.example.test"


def test_endless_redirect_loop_is_bounded(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"images": [{"url": "https://images.example.test/result.png"}]}
            )
        return httpx.Response(
            302, headers={"location": "https://images.example.test/again.png"}
        )

    provider = _editing_provider(handler)

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_loop.png")

    assert "redirected too many times" in str(error.value)


def test_failed_download_parks_the_result_url_for_a_free_retry(tmp_path: Path) -> None:
    """A failed download stays fatal, but the billed result URL must not be lost."""
    result_url = "https://images.example.test/result.png?Signature=abc&Expires=1"
    provider = _editing_provider(
        _serving_handler(result_url, b"<html>sign-in</html>", "text/html")
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_stash.png")

    assert "did not return an image" in str(error.value)
    # The signed URL stays out of the error message, so it cannot reach job records.
    assert result_url not in str(error.value)
    # No half-written picture in the outputs tree.
    assert not (tmp_path / "outputs" / "out_stash.png").exists()

    parked = tmp_path / "recovery" / "out_stash.url.json"
    assert parked.is_file()
    assert parked.stat().st_mode & 0o777 == 0o600
    payload = json.loads(parked.read_text("utf-8"))
    assert payload["resultUrl"] == result_url
    assert payload["model"] == IMAGE_MODEL


def test_successful_download_parks_nothing(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler(
            "https://images.example.test/result.png",
            make_image_bytes(size=(16, 16)),
            "image/png",
        )
    )

    _run_edit(provider, tmp_path, "out_clean.png")

    assert not (tmp_path / "recovery").exists()


def test_unsafe_result_url_is_refused_without_parking_it(tmp_path: Path) -> None:
    provider = _editing_provider(
        _serving_handler("http://127.0.0.1:8000/steal.png", b"root:x:0:0", "text/plain")
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        _run_edit(provider, tmp_path, "out_unsafe.png")

    assert "non-public address" in str(error.value)
    # A hostile URL is refused before anything is written anywhere on disk.
    assert not (tmp_path / "recovery").exists()


def test_factory_returns_a_siliconflow_provider_and_keeps_mock_available(
    tmp_path: Path,
) -> None:
    config = _settings(tmp_path, siliconflow_api_key=API_KEY)

    provider = ImageProviderFactory(config).create("siliconflow")

    assert isinstance(provider, SiliconFlowImageProvider)
    assert provider.model_name == IMAGE_MODEL
    assert provider.endpoint == f"{BASE_URL}/images/generations"
    assert isinstance(ImageProviderFactory(config).create("mock"), MockImageProvider)


def _portrait_plan(client: TestClient) -> tuple[str, dict]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
    )
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    return image_id, plans.json()[0]


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/retouch/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def test_full_job_completes_when_the_cdn_mislabels_a_real_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload → analyze → plan → job → result, with the real download path in play."""
    config = _settings(tmp_path, siliconflow_api_key=API_KEY)
    app = create_app(_settings=config)
    app.dependency_overrides[get_storage_service] = lambda: StorageService(config)
    app.dependency_overrides[get_job_store] = lambda: JobStore(config)
    app.dependency_overrides[get_provider_factory] = lambda: ImageProviderFactory(config)
    app.dependency_overrides[get_agent_brain_factory] = lambda: AgentBrainFactory(config)

    result_bytes = make_image_bytes(size=(48, 64))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url.endswith("/images/generations"):
            return httpx.Response(
                200,
                headers={"x-request-id": "req-e2e"},
                json={"images": [{"url": "https://images.example.test/result.png"}]},
            )
        return httpx.Response(
            200,
            content=result_bytes,
            headers={"content-type": "application/octet-stream"},
        )

    _install_mock_transport(monkeypatch, handler)

    try:
        with TestClient(
            app, headers={"X-Agent-Provider": "local", "X-Action-Provider": "siliconflow"}
        ) as client:
            image_id = client.post(
                "/api/photos/upload",
                files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
            ).json()["imageId"]
            analysis = client.post(
                "/api/photos/analyze", json={"imageId": image_id, "domainType": "auto"}
            ).json()
            plan = client.post("/api/retouch/plans", json={"analysis": analysis}).json()[0]
            created = client.post(
                "/api/retouch/jobs",
                json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
            )
            assert created.status_code == 202
            job = _wait_for_job(client, created.json()["jobId"])
    finally:
        app.dependency_overrides.clear()

    assert analysis["domainType"] == "portrait"
    assert job["status"] == "succeeded", job.get("errorMessage")
    assert len(job["outputImageIds"]) == 1

    outputs = config.data_dir / "outputs"
    written = outputs / f"{job['outputImageIds'][0]}.png"
    assert written.read_bytes() == result_bytes
    metadata_text = (outputs / "_meta" / f"{written.name}.json").read_text("utf-8")
    assert json.loads(metadata_text)["contentType"] == "application/octet-stream"
    assert API_KEY not in metadata_text
    assert seen == [
        f"{BASE_URL}/images/generations",
        "https://images.example.test/result.png",
    ]


def test_api_rejects_siliconflow_actions_without_a_key(client: TestClient) -> None:
    image_id, plan = _portrait_plan(client)

    response = client.post(
        "/api/retouch/jobs",
        headers={"X-Agent-Provider": "local", "X-Action-Provider": "siliconflow"},
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 400
    assert "SILICONFLOW_API_KEY" in response.json()["detail"]
    assert "secret" not in response.text.lower()


def test_portrait_and_landscape_routing_still_hold_with_siliconflow_configured(
    tmp_path: Path,
) -> None:
    config = _settings(tmp_path, siliconflow_api_key=API_KEY)
    app = create_app(_settings=config)
    app.dependency_overrides[get_storage_service] = lambda: StorageService(config)
    app.dependency_overrides[get_job_store] = lambda: JobStore(config)
    app.dependency_overrides[get_provider_factory] = lambda: ImageProviderFactory(config)
    app.dependency_overrides[get_agent_brain_factory] = lambda: AgentBrainFactory(config)

    with TestClient(
        app, headers={"X-Agent-Provider": "local", "X-Action-Provider": "mock"}
    ) as client:
        portrait_id = client.post(
            "/api/photos/upload",
            files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
        ).json()["imageId"]
        landscape_id = client.post(
            "/api/photos/upload",
            files={"file": ("landscape.png", make_landscape_bytes(), "image/png")},
        ).json()["imageId"]

        portrait = client.post(
            "/api/photos/analyze", json={"imageId": portrait_id, "domainType": "auto"}
        ).json()
        landscape = client.post(
            "/api/photos/analyze", json={"imageId": landscape_id, "domainType": "auto"}
        ).json()

        assert portrait["domainType"] == "portrait"
        assert landscape["domainType"] == "general"

        portrait_plans = client.post(
            "/api/retouch/plans", json={"analysis": portrait}
        ).json()
        assert [plan["planId"] for plan in portrait_plans] == ["natural", "avatar", "mood"]

    app.dependency_overrides.clear()
