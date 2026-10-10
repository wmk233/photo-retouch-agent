"""The native-mask backend inside the *existing* retouch job flow.

The P2.8 promise is a workflow, not a module: a photo goes in, the mask is built
locally, the masked backend edits only what the mask opens, the kept pixels are
proved to be the original's, and only then is a success status written -- all of
it driven by the job store and dispatcher that were already there.

These tests run the real job pipeline end to end (upload -> analyse -> plan ->
queue -> execute -> poll) with two fakes at the *designed* seams:

* a stub detector/segmenter, so the mask builder runs without onnxruntime;
* a stub provider that behaves like an inpainting model -- it repaints the white
  side and leaves the black side alone -- without any network call.

Nothing here reaches fal.ai, and no billed request is ever constructed.
"""

from __future__ import annotations

import dataclasses
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import (
    get_agent_brain_factory,
    get_job_store,
    get_provider_factory,
    get_retouch_service,
    get_storage_service,
)
from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.main import create_app
from app.services.job_store import JobStore
from app.services.native_mask_inpainting import NativeMaskInpainting
from app.services.retouch_service import RetouchService
from app.services.storage import StorageService
from tests.test_semantic_edit_mask import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    _image,
    _labels,
    _landmarks,
)

#: The stub model's "new hair": far from any skin tone, so a kept pixel that was
#: repainted cannot hide inside the noise floor of the comparison.
NEW_COLOUR = (18, 24, 48)


class StubNativeMaskProvider:
    """Stands in for fal.ai Qwen inpaint without touching the network.

    ``repaint_style`` selects how honest the fake model is:

    ``masked``
        Repaints the white side and leaves the black side untouched -- a well
        behaved inpainting result.
    ``everything``
        Repaints the whole frame, black side included -- the exact failure the
        pixel-identity check exists to catch.
    """

    provider_name = "fal-inpaint"
    model_name = "fal-ai/qwen-image-edit/inpaint"
    output_extension = ".png"
    requires_edit_mask = True

    def __init__(self, repaint_style: str = "masked") -> None:
        self.repaint_style = repaint_style
        self.calls: list[dict[str, Any]] = []

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: Any,
        user_instruction: str,
        mask_path: Path | None = None,
    ) -> None:
        self.calls.append(
            {
                "sourcePath": Path(source_path),
                "outputPath": Path(output_path),
                "maskPath": Path(mask_path) if mask_path is not None else None,
                "planId": getattr(plan, "plan_id", None),
                "userInstruction": user_instruction,
            }
        )
        source = np.asarray(Image.open(source_path).convert("RGB")).copy()
        mask = np.asarray(Image.open(mask_path).convert("L"))
        assert mask.shape == source.shape[:2], "mask is not in the photo's coordinates"

        painted = np.zeros_like(source)
        painted[:, :] = NEW_COLOUR
        if self.repaint_style == "masked":
            editable = (mask == 255)[..., None]
            result = np.where(editable, painted, source)
        elif self.repaint_style == "everything":
            result = painted
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unknown repaint style: {self.repaint_style}")
        Image.fromarray(result.astype(np.uint8)).save(output_path)


class StubProviderFactory:
    """A provider factory that hands out the stub regardless of the header."""

    def __init__(self, provider: StubNativeMaskProvider) -> None:
        self.provider = provider

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> StubNativeMaskProvider:
        return self.provider

    def capabilities(self) -> dict[str, Any]:
        return {"actionProviders": [self.provider.provider_name], "stub": True}


class _StubDetector:
    """Returns a face mesh scaled to the frame; no MediaPipe, no model file."""

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        mesh = _landmarks().copy()
        mesh[:, 0] *= width / IMAGE_WIDTH
        mesh[:, 1] *= height / IMAGE_HEIGHT
        return [mesh]

    def close(self) -> None:
        return None


class _StubSegmenter:
    """Returns the parsed ATR label map; no onnxruntime, no model file."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        return _labels()


def _upload_source(client: TestClient) -> str:
    buffer = BytesIO()
    Image.fromarray(_image()).save(buffer, format="PNG")
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", buffer.getvalue(), "image/png")},
    )
    assert upload.status_code == 200
    return upload.json()["imageId"]


def _analysis_and_plan(client: TestClient) -> tuple[str, dict]:
    image_id = _upload_source(client)
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    assert analysis.status_code == 200
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    assert plans.status_code == 200
    return image_id, plans.json()[0]


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/retouch/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


@pytest.fixture()
def build_native_client(
    temp_settings: Settings, tmp_path: Path
) -> Callable[..., tuple[TestClient, Settings]]:
    """Create an app whose retouch backend is the stub native-mask provider."""
    opened: list[tuple[TestClient, Any]] = []

    def _build(
        provider: StubNativeMaskProvider,
        *,
        parsing_model: Path | None = None,
    ) -> tuple[TestClient, Settings]:
        model_path = tmp_path / "schp-stub.onnx"
        if parsing_model is None:
            model_path.write_bytes(b"stub")
            parsing_model = model_path
        landmark_path = tmp_path / "face_landmarker.task"
        if not landmark_path.exists():
            landmark_path.write_bytes(b"stub")

        settings = dataclasses.replace(
            temp_settings,
            human_parsing_model_path=parsing_model,
            face_landmark_model_path=landmark_path,
        )
        native = NativeMaskInpainting(
            settings,
            detector_factory=lambda path: _StubDetector(),
            segmenter_factory=lambda path: _StubSegmenter(),
        )
        storage = StorageService(settings)
        jobs = JobStore(settings)
        app = create_app(_settings=settings)
        app.dependency_overrides[get_storage_service] = lambda: storage
        app.dependency_overrides[get_job_store] = lambda: jobs
        app.dependency_overrides[get_provider_factory] = lambda: StubProviderFactory(
            provider
        )
        app.dependency_overrides[get_agent_brain_factory] = lambda: AgentBrainFactory(
            settings
        )
        app.dependency_overrides[get_retouch_service] = lambda: RetouchService(
            storage, jobs, settings=settings, native_mask=native
        )
        client = TestClient(
            app,
            headers={"X-Agent-Provider": "local", "X-Action-Provider": "mock"},
        )
        client.__enter__()
        opened.append((client, app))
        return client, settings

    yield _build

    for client, app in opened:
        client.__exit__(None, None, None)
        app.dependency_overrides.clear()


def _run_job(client: TestClient, image_id: str, plan: dict) -> dict:
    response = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )
    assert response.status_code == 202
    return _wait_for_job(client, response.json()["jobId"])


# --------------------------------------------------------------------- success


def test_native_mask_job_edits_only_the_masked_pixels(
    build_native_client: Callable[..., tuple[TestClient, Settings]],
) -> None:
    provider = StubNativeMaskProvider()
    client, settings = build_native_client(provider)
    image_id, plan = _analysis_and_plan(client)

    job = _run_job(client, image_id, plan)

    assert job["status"] == "succeeded"
    assert job["errorMessage"] is None
    assert job["modelProvider"] == "fal-inpaint"
    assert len(provider.calls) == 1, "the paid edit must happen exactly once"

    call = provider.calls[0]
    assert call["maskPath"] is not None
    assert call["maskPath"].exists()
    assert call["maskPath"] != call["sourcePath"], "image and mask are two inputs"

    source = np.asarray(Image.open(call["sourcePath"]).convert("RGB"))
    mask = np.asarray(Image.open(call["maskPath"]).convert("L"))
    assert mask.shape == source.shape[:2], "the mask is the photo's own size"
    assert set(np.unique(mask)) <= {0, 255}, "the mask is strictly black/white"
    assert 0 < int((mask == 255).sum()) < mask.size, "the mask opens part of the frame"

    native = job["nativeMask"]
    assert native["backend"] == "fal-inpaint"
    assert native["model"] == "fal-ai/qwen-image-edit/inpaint"
    assert native["mask"]["acceptance"]["passed"] is True
    assert native["mask"]["identityCoversEditable"] is True
    assert native["verification"]["mask"]["identityGuardOverlap"] == 0

    statuses = native["verification"]["statuses"]
    assert statuses == {
        "MASK_VALID": True,
        "PIXEL_IDENTITY_PASSED": True,
        "VISUAL_ACCEPTANCE_PENDING": True,
    }
    assert native["verification"]["visualAcceptancePending"] is True

    # The published image is the composite: original pixels where the mask is
    # black, the model's paint where it is white.
    assert len(job["outputImageIds"]) == 1
    output = Path(settings.outputs_dir / f"{job['outputImageIds'][0]}.png")
    assert output.exists()
    final = np.asarray(Image.open(output).convert("RGB"))
    kept = mask == 0
    assert np.array_equal(final[kept], source[kept]), "kept pixels changed"
    edited = mask == 255
    assert np.array_equal(final[edited], np.tile(NEW_COLOUR, (int(edited.sum()), 1)))

    # The record survives the job store's JSON round trip.
    stored = (settings.jobs_dir / f"{job['jobId']}.json").read_text(encoding="utf-8")
    assert '"native_mask"' in stored and "PIXEL_IDENTITY_PASSED" in stored


def test_native_mask_job_saves_and_reloads_the_output(
    build_native_client: Callable[..., tuple[TestClient, Settings]],
) -> None:
    provider = StubNativeMaskProvider()
    client, _ = build_native_client(provider)
    image_id, plan = _analysis_and_plan(client)

    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": "换个发色"},
    ).json()
    job = _wait_for_job(client, created["jobId"])

    refetched = client.get(f"/api/retouch/jobs/{created['jobId']}").json()
    assert refetched["status"] == "succeeded"
    assert refetched["outputImageIds"] == job["outputImageIds"]
    assert refetched["nativeMask"]["verification"]["statuses"] == job["nativeMask"][
        "verification"
    ]["statuses"]
    listed = client.get("/api/retouch/jobs").json()
    assert any(item["jobId"] == job["jobId"] for item in listed)


# --------------------------------------------------------------------- refusal


def test_mask_that_cannot_be_built_fails_without_calling_the_model(
    build_native_client: Callable[..., tuple[TestClient, Settings]],
    tmp_path: Path,
) -> None:
    provider = StubNativeMaskProvider()
    client, settings = build_native_client(
        provider, parsing_model=tmp_path / "schp-missing.onnx"
    )
    image_id, plan = _analysis_and_plan(client)

    job = _run_job(client, image_id, plan)

    assert job["status"] == "failed"
    assert job["outputImageIds"] == []
    assert provider.calls == [], "no photo may leave the machine without a mask"
    assert "human-parsing model is missing" in job["errorMessage"]
    # The intended backend is still recorded, so the failure is not silent.
    assert job["nativeMask"]["backend"] == "fal-inpaint"
    assert job["nativeMask"]["maskPath"].endswith(".png")
    assert not list(settings.outputs_dir.glob("out_*.png"))


def test_repainted_kept_pixels_are_refused_and_never_published(
    build_native_client: Callable[..., tuple[TestClient, Settings]],
) -> None:
    provider = StubNativeMaskProvider(repaint_style="everything")
    client, settings = build_native_client(provider)
    image_id, plan = _analysis_and_plan(client)

    job = _run_job(client, image_id, plan)

    assert job["status"] == "failed"
    assert job["outputImageIds"] == []
    assert job["outputUrls"] == []
    assert "PIXEL_IDENTITY_PASSED" in job["errorMessage"]

    verification = job["nativeMask"]["verification"]
    assert verification["statuses"]["MASK_VALID"] is True
    assert verification["statuses"]["PIXEL_IDENTITY_PASSED"] is False
    assert verification["statuses"]["VISUAL_ACCEPTANCE_PENDING"] is True
    assert verification["aligned"] is False
    assert verification["restoredPath"] is None

    # Neither the composite nor a success record was written.
    assert not list(settings.outputs_dir.glob("out_*.png"))
    stored = (settings.jobs_dir / f"{job['jobId']}.json").read_text(encoding="utf-8")
    assert '"status":"failed"' in stored.replace(" ", "")


def test_native_backend_never_falls_back_to_a_whole_image_edit(
    build_native_client: Callable[..., tuple[TestClient, Settings]],
    tmp_path: Path,
) -> None:
    """A mask failure is a failure, not a silently unmasked generation."""
    provider = StubNativeMaskProvider()
    client, _ = build_native_client(
        provider, parsing_model=tmp_path / "schp-missing.onnx"
    )
    image_id, plan = _analysis_and_plan(client)

    job = _run_job(client, image_id, plan)

    assert job["status"] == "failed"
    assert provider.calls == []
    assert "SiliconFlow" in job["errorMessage"] or "掩膜" in job["errorMessage"]


# ------------------------------------------------------- untouched old workflow


def test_whole_image_provider_keeps_the_old_workflow(client: TestClient) -> None:
    """The mock/SiliconFlow path is unchanged: no mask is involved at all."""
    image_id, plan = _analysis_and_plan(client)

    response = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )
    assert response.status_code == 202
    job = _wait_for_job(client, response.json()["jobId"])

    assert job["status"] == "succeeded"
    assert job["nativeMask"] is None
    assert len(job["outputImageIds"]) == 1
