"""Analysis reuse cache: the fix for the doubled Vision bill.

Clicking "AI design three looks" used to run ``POST /photos/analyze`` and
``POST /styling/proposals`` back to back. Both are billable vision calls, but a
repeat design attempt re-uploaded the same photo and paid for the analysis
again. These tests pin the fix:

* a finished analysis is reused verbatim for identical image bytes + model, so
  the provider is not billed twice;
* the reused result still says ``analysis_reused`` so the UI can report the
  honest number of paid calls;
* nothing about the person is invented -- the cached analysis is the provider's
  own earlier output, only re-bound to the new upload id;
* a different model, a different image, or a brain that cannot describe itself
  never reuses a stored result.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application.commands import AnalyzePhotoCommand
from app.application.use_cases import AnalyzePhoto
from app.domain.analysis import PhotoAnalysis, SubjectSummary
from app.infrastructure.analysis_cache import InMemoryAnalysisCache
from tests.conftest import make_landscape_bytes, make_portrait_bytes


# ------------------------------------------------------------------- test doubles


class _FakeAssets:
    def __init__(self, paths: dict[str, Path]) -> None:
        self._paths = paths

    def resolve_path(self, image_id: str) -> Path | None:
        return self._paths.get(image_id)


class _CountingCatalog:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        self.calls += 1
        return PhotoAnalysis(
            image_id=image_id,
            domain_type="portrait",
            scene_type="portrait",
            subjects=SubjectSummary(count=1, position="center", face_visibility="clear"),
        )


class _CountingBrain:
    """A prompt analyzer that reports how often the (paid) enrich ran."""

    def __init__(self, identity: str | None = "siliconflow:model-a:direct:deadbeef") -> None:
        self.calls = 0
        self._identity = identity

    @property
    def cache_identity(self) -> str | None:
        return self._identity

    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        self.calls += 1
        return PhotoAnalysis(
            image_id=image_id,
            domain_type=baseline.domain_type,
            scene_type=baseline.scene_type,
            subjects=baseline.subjects,
            brain_provider="siliconflow",
            brain_model="model-a",
            vision_mode="direct",
            face_facts={"note": f"provider output #{self.calls}"},
        )


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _run(use_case: AnalyzePhoto, image_id: str, brain: _CountingBrain) -> PhotoAnalysis:
    return asyncio.run(
        use_case.execute(AnalyzePhotoCommand(image_id=image_id, domain_type="auto"), brain)
    )


# ------------------------------------------------------------------- unit: cache


def test_cache_expires_entries_after_the_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryAnalysisCache(ttl_seconds=10.0, max_entries=4)
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "app.infrastructure.analysis_cache.time.monotonic", lambda: clock["now"]
    )

    cache.put("k", _analysis("img_1"))
    assert cache.get("k") is not None

    clock["now"] += 9.9
    assert cache.get("k") is not None

    clock["now"] += 0.2  # now 10.1s old -> expired
    assert cache.get("k") is None
    assert len(cache) == 0


def test_cache_evicts_the_least_recently_used_entry() -> None:
    cache = InMemoryAnalysisCache(ttl_seconds=900.0, max_entries=2)
    cache.put("a", _analysis("img_a"))
    cache.put("b", _analysis("img_b"))
    assert cache.get("a") is not None  # touch "a" so "b" is now the oldest

    cache.put("c", _analysis("img_c"))
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


@pytest.mark.parametrize(
    ("ttl", "max_entries"),
    [(0.0, 8), (-1.0, 8), (60.0, 0), (60.0, -3)],
)
def test_cache_is_disabled_by_non_positive_limits(ttl: float, max_entries: int) -> None:
    cache = InMemoryAnalysisCache(ttl_seconds=ttl, max_entries=max_entries)
    assert cache.enabled is False
    cache.put("k", _analysis("img_1"))
    assert cache.get("k") is None
    assert len(cache) == 0


# ----------------------------------------------------------------- use case: reuse


def test_same_bytes_and_model_reuses_the_analysis_without_a_second_enrich(
    tmp_path: Path,
) -> None:
    first = make_portrait_bytes()
    path_one = _write(tmp_path / "one.png", first)
    path_two = _write(tmp_path / "two.png", first)  # identical bytes, new upload id
    assets = _FakeAssets({"img_one": path_one, "img_two": path_two})
    catalog = _CountingCatalog()
    brain = _CountingBrain()
    use_case = AnalyzePhoto(assets, catalog, cache=InMemoryAnalysisCache())

    fresh = _run(use_case, "img_one", brain)
    assert fresh.analysis_reused is False
    assert brain.calls == 1

    reused = _run(use_case, "img_two", brain)
    # The paid enrich ran exactly once; the second call was served from cache.
    assert brain.calls == 1
    assert reused.analysis_reused is True
    # The provider's own words are reused verbatim, only re-bound to this upload.
    assert reused.image_id == "img_two"
    assert reused.face_facts == fresh.face_facts
    assert reused.face_facts == {"note": "provider output #1"}
    # The cached hit short-circuits before the deterministic baseline too: the
    # stored analysis already carries the identical local measurements.
    assert catalog.calls == 1


def test_a_different_image_never_reuses_the_cache(tmp_path: Path) -> None:
    assets = _FakeAssets(
        {
            "img_a": _write(tmp_path / "a.png", make_portrait_bytes()),
            "img_b": _write(tmp_path / "b.png", make_landscape_bytes()),
        }
    )
    brain = _CountingBrain()
    use_case = AnalyzePhoto(assets, _CountingCatalog(), cache=InMemoryAnalysisCache())

    first = _run(use_case, "img_a", brain)
    second = _run(use_case, "img_b", brain)

    assert brain.calls == 2
    assert second.analysis_reused is False
    assert second.face_facts != first.face_facts


def test_switching_the_model_never_reuses_another_models_result(tmp_path: Path) -> None:
    payload = make_portrait_bytes()
    assets = _FakeAssets(
        {
            "img_one": _write(tmp_path / "one.png", payload),
            "img_two": _write(tmp_path / "two.png", payload),
        }
    )
    cache = InMemoryAnalysisCache()
    use_case = AnalyzePhoto(assets, _CountingCatalog(), cache=cache)

    model_a = _CountingBrain("siliconflow:model-a:direct:aaaa")
    model_b = _CountingBrain("siliconflow:model-b:direct:bbbb")

    _run(use_case, "img_one", model_a)
    result_b = _run(use_case, "img_two", model_b)

    assert model_a.calls == 1
    assert model_b.calls == 1
    assert result_b.analysis_reused is False


def test_a_brain_without_identity_is_never_cached(tmp_path: Path) -> None:
    payload = make_portrait_bytes()
    assets = _FakeAssets(
        {
            "img_one": _write(tmp_path / "one.png", payload),
            "img_two": _write(tmp_path / "two.png", payload),
        }
    )
    brain = _CountingBrain(identity=None)
    cache = InMemoryAnalysisCache()
    use_case = AnalyzePhoto(assets, _CountingCatalog(), cache=cache)

    _run(use_case, "img_one", brain)
    second = _run(use_case, "img_two", brain)

    assert brain.calls == 2
    assert second.analysis_reused is False
    assert len(cache) == 0


def test_without_a_cache_the_use_case_still_runs_the_old_way(tmp_path: Path) -> None:
    payload = make_portrait_bytes()
    assets = _FakeAssets({"img_one": _write(tmp_path / "one.png", payload)})
    brain = _CountingBrain()
    use_case = AnalyzePhoto(assets, _CountingCatalog(), cache=None)

    result = _run(use_case, "img_one", brain)
    assert brain.calls == 1
    assert result.analysis_reused is False


# --------------------------------------------------------------------- api level


def test_analyze_endpoint_reuses_a_cached_analysis_for_the_same_bytes(
    client: TestClient,
) -> None:
    payload = make_portrait_bytes()

    first_upload = client.post(
        "/api/photos/upload", files={"file": ("portrait.png", payload, "image/png")}
    )
    second_upload = client.post(
        "/api/photos/upload", files={"file": ("portrait.png", payload, "image/png")}
    )
    assert first_upload.status_code == 200
    assert second_upload.status_code == 200
    first_id = first_upload.json()["imageId"]
    second_id = second_upload.json()["imageId"]
    assert first_id != second_id  # a fresh upload id, identical pixels

    first = client.post("/api/photos/analyze", json={"imageId": first_id})
    second = client.post("/api/photos/analyze", json={"imageId": second_id})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["analysisReused"] is False
    body = second.json()
    assert body["analysisReused"] is True
    # The reused analysis is bound to the new upload and keeps the same facts.
    assert body["imageId"] == second_id
    assert body["faceFacts"] == first.json()["faceFacts"]
    assert body["domainType"] == "portrait"


def test_styling_providers_report_the_full_three_call_design_flow(
    client: TestClient,
) -> None:
    body = client.get("/api/styling/providers").json()

    assert body["stylingAiAnalysisRequests"] == 1
    assert body["stylingAiDesignRequests"] == 1
    assert body["stylingAiGenerationRequests"] == 1
    assert body["stylingAiFlowRequests"] == 3

    ai_mode = body["stylingDesignModes"]["ai"]
    assert ai_mode["analysisRequests"] == 1
    assert ai_mode["designRequests"] == 1
    assert ai_mode["generationRequests"] == 1
    # The whole flow (analysis + design + generation planning) is three calls,
    # and the cached-analysis path still bills the design and planning passes.
    assert ai_mode["paidRequests"] == 3
    assert ai_mode["reusedAnalysisRequests"] == 2
    assert "1 次人物事实分析" in ai_mode["costNote"]
    assert "3 次付费模型调用" in ai_mode["costNote"]
    # The copy must not imply the text-only model looked at the photo.
    assert "不会看到原图" in ai_mode["costNote"]

    template_mode = body["stylingDesignModes"]["template"]
    assert template_mode["paidRequests"] == 0

    # The English billing note must agree with the Chinese copy.
    assert "3 requests" in body["billingNote"]
    assert "never see the photo" in body["billingNote"]
    assert "free" in body["billingNote"]


def _analysis(image_id: str) -> PhotoAnalysis:
    return PhotoAnalysis(
        image_id=image_id,
        domain_type="portrait",
        scene_type="portrait",
        subjects=SubjectSummary(count=1, position="center", face_visibility="clear"),
    )
