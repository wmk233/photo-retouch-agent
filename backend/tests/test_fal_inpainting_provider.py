"""fal.ai native-mask inpainting provider tests.

Every HTTP call below is served by ``httpx.MockTransport``. Nothing leaves the
machine, no fal queue is touched and no billable request is ever issued. The
tests pin the two properties the P2.8 workflow depends on:

* the request really carries **two different inputs** -- the original photo in
  ``image_url`` and the black/white edit mask in ``mask_url``;
* a missing key, a bad mask or a failed queue run is refused without ever
  sending (or re-sending) a billed request.
"""
from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.config import Settings
from app.providers.factory import ImageProviderFactory
from app.providers.fal_inpainting_provider import (
    FalInpaintingProvider,
    FalInpaintingProviderError,
)
from app.schemas.retouch import RetouchPlan
from tests.conftest import make_image_bytes

ENDPOINT = "https://queue.example.test/fal-ai/qwen-image-edit/inpaint"
QUEUE_BASE = "https://queue.example.test"
RESULT_IMAGE_URL = "https://cdn.example.test/result.png"
API_KEY = "fal-test-key-0000"
REQUEST_ID = "req-fal-0001"

SOURCE_SIZE = (160, 120)


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
        "fal_key": None,
        "face_lock_mode": "off",
        "rate_limit_per_second": 0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _plan(plan_id: str = "natural", prompt: str = "换自然棕色长发并搭配米色针织衫。") -> RetouchPlan:
    return RetouchPlan(
        plan_id=plan_id,
        domain_type="portrait",
        title=plan_id,
        description=plan_id,
        intensity="natural",
        edit_prompt=prompt,
        negative_prompt="避免改变五官",
        expected_changes=["换发型", "换服装"],
    )


def _png_bytes(size: tuple[int, int] = (24, 18), color: tuple[int, int, int] = (200, 40, 90)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _source_and_mask(
    tmp_path: Path,
    *,
    mask_size: tuple[int, int] = SOURCE_SIZE,
    mask_levels: tuple[int, int] = (0, 255),
    editable_box: tuple[int, int, int, int] = (20, 20, 70, 80),
    source_size: tuple[int, int] = SOURCE_SIZE,
) -> tuple[Path, Path]:
    """A photo plus a strictly black/white mask in the photo's own coordinates."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "subject.png"
    image = Image.new("RGB", source_size, color=(180, 190, 205))
    for x in range(source_size[0]):
        image.putpixel((x, 0), (x % 256, 30, 60))
    image.save(source_path)

    mask_path = source_dir / "edit_mask.png"
    keep, paint = mask_levels
    mask = Image.new("L", mask_size, color=keep)
    if paint != keep:
        for y in range(editable_box[1], editable_box[3]):
            for x in range(editable_box[0], editable_box[2]):
                if 0 <= x < mask_size[0] and 0 <= y < mask_size[1]:
                    mask.putpixel((x, y), paint)
    mask.save(mask_path)
    return source_path, mask_path


def _data_uri_bytes(uri: str) -> bytes:
    header, _, encoded = uri.partition(",")
    assert header.startswith("data:image/"), header
    return base64.b64decode(encoded)


def _flow_handler(
    captured: dict,
    *,
    submit_payload: dict | None = None,
    status_payload: dict | None = None,
    result_payload: dict | None = None,
    result_bytes: bytes | None = None,
    result_content_type: str = "image/png",
    status_url_override: str | None = None,
):
    """Serve the whole queue lifecycle: POST -> status GET -> result GET -> download."""
    status_url = status_url_override or f"{QUEUE_BASE}/status/{REQUEST_ID}"
    result_url = f"{QUEUE_BASE}/result/{REQUEST_ID}"
    image_bytes = result_bytes if result_bytes is not None else _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"].append(request)
        url = str(request.url)
        if request.method == "POST" and url == ENDPOINT:
            captured["posts"] += 1
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            captured["headers"] = dict(request.headers)
            body = submit_payload if submit_payload is not None else {
                "request_id": REQUEST_ID,
                "status_url": status_url,
                "response_url": result_url,
            }
            return httpx.Response(200, json=body, headers={"content-type": "application/json"})
        if request.method == "GET" and url == status_url:
            captured["polls"] += 1
            body = status_payload if status_payload is not None else {"status": "COMPLETED"}
            return httpx.Response(200, json=body, headers={"content-type": "application/json"})
        if request.method == "GET" and url == result_url:
            captured["results"] += 1
            body = result_payload if result_payload is not None else {
                "images": [{"url": RESULT_IMAGE_URL}],
                "seed": 4242,
            }
            return httpx.Response(200, json=body, headers={"content-type": "application/json"})
        if request.method == "GET" and url == RESULT_IMAGE_URL:
            captured["downloads"] += 1
            return httpx.Response(
                200, content=image_bytes, headers={"content-type": result_content_type}
            )
        return httpx.Response(404, json={"detail": f"unexpected {request.method} {url}"})

    return handler


def _captured() -> dict:
    return {"requests": [], "posts": 0, "polls": 0, "results": 0, "downloads": 0}


def _provider(captured: dict, **overrides: object) -> FalInpaintingProvider:
    kwargs: dict[str, object] = {
        "api_key": API_KEY,
        "endpoint": ENDPOINT,
        "poll_interval_seconds": 0.0,
        "max_poll_seconds": 5.0,
    }
    handler = overrides.pop("handler", None)
    kwargs.update(overrides)
    transport = httpx.MockTransport(handler or _flow_handler(captured))
    return FalInpaintingProvider(transport=transport, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------- wiring


def test_fal_inpaint_is_registered_and_reports_unconfigured(tmp_path: Path) -> None:
    factory = ImageProviderFactory(_settings(tmp_path))
    capabilities = factory.capabilities()
    ids = [entry["id"] for entry in capabilities["actionProviders"]]
    assert "fal-inpaint" in ids
    entry = next(e for e in capabilities["actionProviders"] if e["id"] == "fal-inpaint")
    assert entry["label"] == "fal.ai Qwen Inpaint"
    assert entry["model"] == "fal-ai/qwen-image-edit/inpaint"
    assert entry["configured"] is False
    assert entry["requiresApiKey"] is True
    assert capabilities["falInpaintConfigured"] is False


def test_fal_inpaint_reports_configured_when_a_key_exists(tmp_path: Path) -> None:
    factory = ImageProviderFactory(_settings(tmp_path, fal_key=API_KEY))
    capabilities = factory.capabilities()
    entry = next(e for e in capabilities["actionProviders"] if e["id"] == "fal-inpaint")
    assert entry["configured"] is True
    assert capabilities["falInpaintConfigured"] is True


@pytest.mark.parametrize("alias", ["fal-inpaint", "fal", "qwen-inpaint"])
def test_fal_aliases_resolve_to_the_inpaint_backend(tmp_path: Path, alias: str) -> None:
    factory = ImageProviderFactory(_settings(tmp_path, fal_key=API_KEY))
    provider = factory.create(alias)
    assert isinstance(provider, FalInpaintingProvider)
    assert provider.provider_name == "fal-inpaint"
    assert provider.model_name == "fal-ai/qwen-image-edit/inpaint"
    assert provider.output_extension == ".png"
    assert provider.requires_edit_mask is True


def test_factory_refuses_without_a_key_and_never_leaks_another_key(tmp_path: Path) -> None:
    factory = ImageProviderFactory(_settings(tmp_path, fal_key=None))
    with pytest.raises(HTTPException) as excinfo:
        factory.create("fal-inpaint")
    assert excinfo.value.status_code == 400
    # The error tells the operator which env var to set, and carries no secret.
    assert "FAL_KEY" in str(excinfo.value.detail)


def test_auto_never_selects_the_fal_backend(tmp_path: Path) -> None:
    factory = ImageProviderFactory(_settings(tmp_path, fal_key=API_KEY))
    provider = factory.create("auto")
    assert getattr(provider, "requires_edit_mask", False) is False


# ------------------------------------------------------------ preconditions


def test_edit_image_requires_an_api_key_before_any_request(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    provider = _provider(captured, api_key="")
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    message = str(excinfo.value)
    assert "no request was sent" in message
    assert "FAL_KEY" in message
    assert captured["posts"] == 0
    assert captured["requests"] == []


def test_edit_image_refuses_without_a_mask(tmp_path: Path) -> None:
    captured = _captured()
    source, _mask = _source_and_mask(tmp_path)
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), ""))
    assert "requires an edit mask" in str(excinfo.value)
    assert captured["posts"] == 0


def test_edit_image_refuses_when_source_image_is_missing(tmp_path: Path) -> None:
    captured = _captured()
    _source, mask = _source_and_mask(tmp_path)
    provider = _provider(captured)
    missing = tmp_path / "nope.png"
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(missing, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "Source image not found" in str(excinfo.value)
    assert captured["posts"] == 0


def test_edit_image_refuses_when_mask_is_missing(tmp_path: Path) -> None:
    captured = _captured()
    source, _mask = _source_and_mask(tmp_path)
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(
            provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=tmp_path / "missing_mask.png")
        )
    assert "Edit mask not found" in str(excinfo.value)
    assert captured["posts"] == 0


# ------------------------------------------------------------ the real request


def test_request_carries_distinct_image_and_mask_inputs(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    output = tmp_path / "out.png"
    provider = _provider(captured)
    asyncio.run(
        provider.edit_image(source, output, _plan(), "刘海不要太厚", mask_path=mask)
    )

    assert captured["posts"] == 1
    payload = captured["payload"]
    assert payload["output_format"] == "png"
    assert payload["num_images"] == 1
    assert payload["sync_mode"] is False

    # The two inputs are separate fields and carry different bytes: the photo
    # and the mask, never one standing in for the other.
    assert payload["image_url"] != payload["mask_url"]
    assert _data_uri_bytes(payload["image_url"]) == source.read_bytes()
    assert _data_uri_bytes(payload["mask_url"]) == mask.read_bytes()
    assert payload["image_url"].startswith("data:image/png;base64,")
    assert payload["mask_url"].startswith("data:image/png;base64,")

    # The prompt is the planner's design plus the user's extra instruction.
    assert "换自然棕色长发并搭配米色针织衫。" in payload["prompt"]
    assert "刘海不要太厚" in payload["prompt"]

    headers = captured["headers"]
    assert headers["authorization"] == f"Key {API_KEY}"
    assert headers["content-type"] == "application/json"


def test_output_bytes_and_metadata_are_persisted(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    output = tmp_path / "outputs" / "abc123.png"
    provider = _provider(captured)
    asyncio.run(provider.edit_image(source, output, _plan(), "", mask_path=mask))

    assert captured["downloads"] == 1
    assert output.is_file()
    assert output.read_bytes() == _png_bytes()

    metadata_path = output.parent / "_meta" / f"{output.name}.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["provider"] == "fal-inpaint"
    assert metadata["model"] == "fal-ai/qwen-image-edit/inpaint"
    assert metadata["mode"] == "native-mask-inpaint"
    assert metadata["inputs"] == ["image_url", "mask_url"]
    assert metadata["requestId"] == REQUEST_ID
    assert metadata["mask"]["editablePixels"] > 0
    assert metadata["mask"]["semantics"] == "white=repaint black=keep original pixels"
    assert metadata["planId"] == "natural"
    # No signed URL and no credentials are persisted.
    assert API_KEY not in metadata_path.read_text(encoding="utf-8")


# --------------------------------------------------------------- mask gate


def test_mask_size_mismatch_is_refused_before_any_request(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path, mask_size=(80, 60))
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "own coordinates" in str(excinfo.value)
    assert captured["posts"] == 0


def test_grey_mask_is_refused_before_any_request(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path, mask_levels=(0, 128))
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "strictly black/white" in str(excinfo.value)
    assert captured["posts"] == 0


def test_all_black_mask_is_refused_before_any_request(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path, mask_levels=(0, 0))
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "almost nothing" in str(excinfo.value)
    assert captured["posts"] == 0


def test_whole_image_mask_is_refused_before_any_request(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path, mask_levels=(255, 255))
    provider = _provider(captured)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "whole-image edit" in str(excinfo.value)
    assert captured["posts"] == 0


# --------------------------------------------------------------- failure paths


def test_failed_status_is_reported_and_not_resubmitted(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    provider = _provider(
        captured,
        handler=_flow_handler(captured, status_payload={"status": "FAILED", "detail": "model oom"}),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "FAILED" in str(excinfo.value)
    assert "model oom" in str(excinfo.value)
    # Exactly one billed POST happened, and no second one was issued.
    assert captured["posts"] == 1
    assert captured["downloads"] == 0


def test_submit_transport_error_is_not_retried(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"].append(request)
        if request.method == "POST":
            captured["posts"] += 1
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(404)

    provider = _provider(captured, handler=handler)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "not retried" in str(excinfo.value)
    assert captured["posts"] == 1


def test_poll_timeout_is_not_resubmitted(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    provider = _provider(
        captured,
        poll_interval_seconds=0.01,
        max_poll_seconds=0.001,
        handler=_flow_handler(captured, status_payload={"status": "IN_QUEUE"}),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    message = str(excinfo.value)
    assert "NOT" in message
    assert REQUEST_ID in message
    assert captured["posts"] == 1


def test_status_url_on_a_foreign_host_is_refused(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    provider = _provider(
        captured,
        handler=_flow_handler(captured, status_url_override="https://evil.example.test/status/1"),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    assert "unexpected host" in str(excinfo.value)
    assert captured["posts"] == 1
    assert captured["polls"] == 0


def test_non_image_download_is_refused(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    output = tmp_path / "out.png"
    provider = _provider(
        captured,
        handler=_flow_handler(
            captured,
            result_bytes=b"<html>sign in to continue</html>",
            result_content_type="text/html",
        ),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, output, _plan(), "", mask_path=mask))
    assert "did not return an image" in str(excinfo.value)
    assert not output.exists()


def test_undecodable_image_bytes_are_refused(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    output = tmp_path / "out.png"
    provider = _provider(
        captured,
        handler=_flow_handler(
            captured,
            result_bytes=b"not really a png",
            result_content_type="image/png",
        ),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, output, _plan(), "", mask_path=mask))
    assert "could not be decoded" in str(excinfo.value)
    assert not output.exists()


def test_error_detail_redacts_the_api_key(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"].append(request)
        if request.method == "POST":
            captured["posts"] += 1
            return httpx.Response(
                401,
                json={"detail": f"Invalid API key {API_KEY}"},
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404)

    provider = _provider(captured, handler=handler)
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    message = str(excinfo.value)
    assert "[redacted]" in message
    assert API_KEY not in message
    assert captured["posts"] == 1


def test_result_without_an_image_is_refused(tmp_path: Path) -> None:
    captured = _captured()
    source, mask = _source_and_mask(tmp_path)
    provider = _provider(
        captured,
        handler=_flow_handler(captured, result_payload={"images": []}),
    )
    with pytest.raises(FalInpaintingProviderError) as excinfo:
        asyncio.run(provider.edit_image(source, tmp_path / "out.png", _plan(), "", mask_path=mask))
    # An empty images list does not look like output, so the queued request is
    # followed to its response_url, which also carries nothing usable.
    assert "did not contain an output image" in str(excinfo.value)
    assert captured["posts"] == 1
    assert captured["downloads"] == 0


def test_make_image_bytes_helper_is_compatible_with_the_provider(tmp_path: Path) -> None:
    """Sanity check: the shared test image helper survives the provider path."""
    captured = _captured()
    source = tmp_path / "helper.png"
    source.write_bytes(make_image_bytes(size=SOURCE_SIZE))
    mask = tmp_path / "helper_mask.png"
    mask_image = Image.new("L", SOURCE_SIZE, color=0)
    for y in range(30, 90):
        for x in range(30, 90):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)
    provider = _provider(captured)
    output = tmp_path / "helper_out.png"
    asyncio.run(provider.edit_image(source, output, _plan(), "", mask_path=mask))
    assert output.is_file()
    assert captured["posts"] == 1
