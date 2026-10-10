"""Unit tests for the RunningHub CropAndStitch provider.

Nothing here touches the network: ``rh.upload`` / ``rh.create_task`` /
``rh.poll_task`` / ``rh.fetch_outputs`` / ``rh.download`` are replaced with local
fakes, so the real ``build_workflow``, ``validate_workflow`` and
``collect_gate_checks`` still run and the billed ``create_task`` is provably
issued at most once.

The properties pinned here are the ones the job layer relies on:

* the semantic mask *set* is mandatory -- no union-mask fallback;
* the prompts come from the plan the user selected (never a hard-coded style);
* a refused preflight is a normal provider error, not a ``SystemExit`` that
  would escape ``execute_job``'s ``except Exception``;
* an ambiguous create call is reported, never retried.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from app.infrastructure import runninghub_crop_stitch as rh
from app.providers.runninghub_crop_stitch_provider import (
    PACKAGE_MASKS,
    RunningHubCropStitchProvider,
    RunningHubCropStitchProviderError,
    _refusal_message,
)
from app.schemas.retouch import RetouchPlan

WIDTH, HEIGHT = 200, 260

EDIT_PROMPT = (
    "Restyle the person in the uploaded photo into the selected look. "
    "Edit this photo directly; keep the same person.\n"
    "Hair: change the hair from the person's current shoulder-length hair to "
    "shoulder-past long hair with soft large waves, and change the hair colour "
    "to 酒红棕 (Burgundy Wine Brown).\n"
    "Outfit: replace the current top with a deep olive-green silk blouse with a "
    "high closed collar, cut from the same deep olive-green cloth.\n"
)


def _plan(plan_id: str = "french_vintage", *, edit_prompt: str = EDIT_PROMPT) -> RetouchPlan:
    return RetouchPlan(
        plan_id=plan_id,
        domain_type="portrait",
        title=f"look-{plan_id}",
        description="test look",
        intensity="medium",
        edit_prompt=edit_prompt,
        negative_prompt="avoid changing the face",
        expected_changes=["hair", "outfit"],
    )


class _MaskSet:
    """Stand-in for ``EditMaskSet`` with the three attributes the packager reads."""

    def __init__(self, *, hair=None, clothing=None, protected=None) -> None:
        self.hair = hair
        self.clothing = clothing
        self.protected = protected


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return inside.astype(np.uint8) * 255


def _mask_set(**overrides: np.ndarray) -> _MaskSet:
    """A geometry the real preflight gate accepts: hair/clothing away from the core."""
    hair = np.zeros((HEIGHT, WIDTH), np.uint8)
    hair[0:30, :] = 255
    hair[30:150, 0:58] = 255
    hair[30:150, 142:WIDTH] = 255
    clothing = np.zeros((HEIGHT, WIDTH), np.uint8)
    clothing[190:HEIGHT, :] = 255
    values = {
        "hair": hair,
        "clothing": clothing,
        "protected": _ellipse(WIDTH / 2, 90, 40, 55),
    }
    values.update(overrides)
    return _MaskSet(**values)


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "inputs" / "subject.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), (180, 150, 140))
    for x in range(WIDTH):
        image.putpixel((x, 0), (x % 256, 30, 60))
    image.save(path)
    return path


def _provider(tmp_path: Path, **overrides) -> RunningHubCropStitchProvider:
    kwargs = {
        "api_key": "unit-test-key-0123456789",
        "object_info_path": None,
        "key_file": tmp_path / "missing.runninghub_key",
    }
    kwargs.update(overrides)
    return RunningHubCropStitchProvider(**kwargs)


class _FakeNetwork:
    """Records the mocked platform calls; only ``create_task`` may be billed."""

    def __init__(self) -> None:
        self.calls = {"upload": 0, "create_task": 0, "poll_task": 0, "download": 0}
        self.created_workflow: dict | None = None
        self.create_error: Exception | None = None
        self.poll_status = "SUCCESS"

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def upload(client, key, path):
            self.calls["upload"] += 1
            return f"uploaded-{Path(path).name}"

        def create_task(client, key, workflow, *, workflow_id, instance_type):
            self.calls["create_task"] += 1
            self.created_workflow = workflow
            if self.create_error is not None:
                raise self.create_error
            return {"data": {"taskId": "task-unit-0001"}}

        def poll_task(client, key, task_id, *, timeout_seconds, interval_seconds, on_status=None):
            self.calls["poll_task"] += 1
            if on_status is not None:
                on_status("RUNNING")
                on_status(self.poll_status)
            return self.poll_status

        def fetch_outputs(client, key, task_id):
            return [
                {"fileUrl": "https://cdn.test/clothing.png"},
                {"fileUrl": "https://cdn.test/final.png"},
            ]

        def download(client, url):
            self.calls["download"] += 1
            return b"\x89PNG\r\n\x1a\nfake"

        for name, function in (
            ("upload", upload),
            ("create_task", create_task),
            ("poll_task", poll_task),
            ("fetch_outputs", fetch_outputs),
            ("download", download),
        ):
            monkeypatch.setattr(rh, name, function)


# ------------------------------------------------------------------ contract


def test_provider_declares_the_mask_set_contract() -> None:
    assert RunningHubCropStitchProvider.provider_name == "runninghub-crop-stitch"
    assert RunningHubCropStitchProvider.requires_edit_mask is True
    assert RunningHubCropStitchProvider.accepts_mask_set is True
    assert RunningHubCropStitchProvider.output_extension == ".png"


def test_missing_mask_set_is_refused_without_any_request(tmp_path: Path, monkeypatch) -> None:
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = _provider(tmp_path)

    with pytest.raises(RunningHubCropStitchProviderError, match="mask set"):
        asyncio.run(
            provider.edit_image(
                _source(tmp_path), tmp_path / "out" / "final.png", _plan(), ""
            )
        )

    assert network.calls == {"upload": 0, "create_task": 0, "poll_task": 0, "download": 0}
    assert provider.paid_requests == 0


def test_missing_edit_prompt_is_refused(tmp_path: Path, monkeypatch) -> None:
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = _provider(tmp_path)

    with pytest.raises(RunningHubCropStitchProviderError, match="edit prompt"):
        provider._run_sync(
            _source(tmp_path), tmp_path / "out" / "final.png", _plan(edit_prompt=""), _mask_set()
        )

    assert network.calls["create_task"] == 0


def test_unusable_plan_prompt_becomes_a_normal_provider_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A ``SystemExit`` inside the builder must not escape ``execute_job``."""
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = _provider(tmp_path)
    useless = "Outfit: a plain garment with no target marker at all.\n"

    # The raw marker wording is replaced with an actionable message that tells
    # the user this provider only takes styling plans and that nothing was
    # billed -- the refusal is guaranteed to precede ``create_task``.
    with pytest.raises(RunningHubCropStitchProviderError, match="未产生任何费用"):
        provider._run_sync(
            _source(tmp_path),
            tmp_path / "out" / "final.png",
            _plan(edit_prompt=useless),
            _mask_set(),
        )

    assert network.calls["create_task"] == 0


def test_refusal_message_names_the_wrong_plan_and_the_zero_charge() -> None:
    message = _refusal_message(
        SystemExit("clothing: plan prompt has no target marker to extract")
    )

    assert "AI 造型设计" in message
    assert "皮肤美化" in message
    assert "未产生任何费用" in message
    # The raw internal marker wording must not leak to the user.
    assert "target marker" not in message


def test_missing_key_is_refused_before_the_package_is_built(tmp_path: Path, monkeypatch) -> None:
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = RunningHubCropStitchProvider(
        api_key=None, object_info_path=None, key_file=tmp_path / "absent.key"
    )

    with pytest.raises(RunningHubCropStitchProviderError, match="API key"):
        provider._run_sync(
            _source(tmp_path), tmp_path / "out" / "final.png", _plan(), _mask_set()
        )

    assert network.calls["create_task"] == 0


# ------------------------------------------------------------------- prompts


@pytest.mark.parametrize("plan_id", ["french_vintage", "artistic_wave", "ai_elegant_refined"])
def test_prompts_are_built_from_the_selected_plan(tmp_path: Path, plan_id: str) -> None:
    """No style is hard-coded: the plan id drives ``expected_style_id``."""
    provider = _provider(tmp_path)
    _bundle, prompts = provider._build_prompts(_plan(plan_id), plan_id)

    assert prompts["style_id"] == plan_id
    assert "shoulder-past long hair" in prompts["hair"]
    assert "olive-green" in prompts["clothing"]


def test_two_plans_produce_their_own_prompts(tmp_path: Path) -> None:
    """The selected plan's own text reaches the graph, not a fixed look."""
    provider = _provider(tmp_path)
    first_prompt = (
        "Outfit: replace the current top with a deep olive-green silk blouse.\n"
        "Hair: change the hair from the current shoulder-length hair to a short "
        "blunt bob, and change the hair colour to 冷调灰棕 (Ash Greige).\n"
    )
    second_prompt = (
        "Outfit: replace the current top with a cream linen shirt.\n"
        "Hair: change the hair from the current shoulder-length hair to long "
        "curtain-bang waves, and change the hair colour to 酒红棕 (Burgundy Wine Brown).\n"
    )
    first = provider._build_prompts(_plan("look_one", edit_prompt=first_prompt), "look_one")[1]
    second = provider._build_prompts(_plan("look_two", edit_prompt=second_prompt), "look_two")[1]

    assert first["style_id"] == "look_one"
    assert second["style_id"] == "look_two"
    assert "olive-green silk blouse" in first["clothing"]
    assert "cream linen shirt" in second["clothing"]
    assert "olive-green silk blouse" not in second["clothing"]
    assert "blunt bob" in first["hair"]
    assert "blunt bob" not in second["hair"]


# ------------------------------------------------------------------- package


def test_write_package_stages_binary_masks_and_manifest(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    mask_set = _mask_set(hair=np.full((HEIGHT, WIDTH), 200, np.uint8))

    size = provider._write_package(package, _source(tmp_path), mask_set)

    assert size == (WIDTH, HEIGHT)
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["source_image"] == {"width": WIDTH, "height": HEIGHT}

    for name in PACKAGE_MASKS:
        with Image.open(package / name) as image:
            assert image.mode == "L"
            assert image.size == (WIDTH, HEIGHT)
            assert set(np.unique(np.asarray(image))) <= {0, 255}

    hair = np.asarray(Image.open(package / "hair_edit_mask.png"))
    assert set(np.unique(hair)) == {255}  # 200 -> white, never a grey pass-through

    with Image.open(package / "original.jpg") as original:
        assert original.mode == "RGB"
        assert original.size == (WIDTH, HEIGHT)


@pytest.mark.parametrize("attribute", ["hair", "clothing", "protected"])
def test_write_package_rejects_a_mismatched_mask_size(tmp_path: Path, attribute: str) -> None:
    provider = _provider(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    broken = np.zeros((HEIGHT - 10, WIDTH), np.uint8)

    with pytest.raises(RunningHubCropStitchProviderError, match="refusing to resize"):
        provider._write_package(package, _source(tmp_path), _mask_set(**{attribute: broken}))


# ----------------------------------------------------------- full mocked flow


def test_edit_image_creates_exactly_one_billed_task(tmp_path: Path, monkeypatch) -> None:
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = _provider(tmp_path)
    output = tmp_path / "outputs" / "final.png"
    stages: list[str] = []
    provider.on_progress = stages.append

    asyncio.run(
        provider.edit_image(_source(tmp_path), output, _plan("look_selected"), "请改造", mask_set=_mask_set())
    )

    assert network.calls["upload"] == 3
    assert network.calls["create_task"] == 1
    assert network.calls["poll_task"] == 1
    assert network.calls["download"] == 2

    assert provider.paid_requests == 1
    assert provider.last_task_id == "task-unit-0001"
    assert output.is_file()
    assert provider.last_clothing_pass_path is not None
    assert provider.last_clothing_pass_path.name == "final_clothing_pass.png"
    assert provider.last_clothing_pass_path.is_file()

    # The graph the platform received carries the selected plan's prompts.
    assert provider.last_prompts is not None
    assert provider.last_prompts["style_id"] == "look_selected"
    serialised = json.dumps(network.created_workflow)
    assert "look_selected" not in serialised  # style id lives in prompts, not the graph
    assert "olive-green" in serialised

    # Progress is reported in the stages the UI shows, and only once per phase.
    assert "provider_preflight" in stages
    assert stages.count("provider_upload") == 1
    assert "provider_submitted" in stages
    assert "provider_download" in stages

    # Audit trail: gate and meta sit beside the frame, with no credentials.
    meta_dir = output.parent / "_meta"
    for kind in ("gate", "workflow", "task", "meta"):
        assert (meta_dir / f"final.png.{kind}.json").is_file()
    meta = json.loads((meta_dir / "final.png.meta.json").read_text())
    assert meta["styleIdMatchesPlan"] is True
    assert meta["paidRequests"] == 1
    assert "unit-test-key" not in json.dumps(meta)


def test_gate_rejection_stops_before_the_billed_call(tmp_path: Path, monkeypatch) -> None:
    """A mask that reaches the facial core must abort at the gate, unpaid."""
    network = _FakeNetwork()
    network.install(monkeypatch)
    provider = _provider(tmp_path)
    mask_set = _mask_set(clothing=np.full((HEIGHT, WIDTH), 255, np.uint8))

    with pytest.raises(RunningHubCropStitchProviderError, match="preflight gate rejected"):
        provider._run_sync(
            _source(tmp_path), tmp_path / "out" / "final.png", _plan(), mask_set
        )

    assert network.calls["create_task"] == 0
    assert network.calls["upload"] == 0
    assert provider.paid_requests == 0


def test_ambiguous_create_call_is_an_error_and_is_never_retried(
    tmp_path: Path, monkeypatch
) -> None:
    network = _FakeNetwork()
    network.create_error = httpx.ConnectError("connection reset")
    network.install(monkeypatch)
    provider = _provider(tmp_path)

    with pytest.raises(RunningHubCropStitchProviderError, match="NOT retried"):
        provider._run_sync(_source(tmp_path), tmp_path / "out" / "final.png", _plan(), _mask_set())

    assert network.calls["create_task"] == 1
    assert network.calls["poll_task"] == 0


def test_failed_task_does_not_retry_the_create_call(tmp_path: Path, monkeypatch) -> None:
    network = _FakeNetwork()
    network.poll_status = "FAILED"
    network.install(monkeypatch)
    provider = _provider(tmp_path)

    with pytest.raises(rh.RunningHubError):
        provider._run_sync(_source(tmp_path), tmp_path / "out" / "final.png", _plan(), _mask_set())

    assert network.calls["create_task"] == 1
    assert network.calls["download"] == 0
