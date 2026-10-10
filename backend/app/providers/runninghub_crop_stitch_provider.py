"""RunningHub CropAndStitch provider: the validated two-pass masked edit.

This is the web app's face of :mod:`app.infrastructure.runninghub_crop_stitch`,
the module that produced the accepted frame for RunningHub task
``2108330225116082177``. The CLI (``tools/runninghub_crop_stitch_run.py``) and
this provider therefore drive *the same* graph, prompt builder and gate checks;
nothing about the run is re-implemented here.

The provider implements the shared ``ImageEditProvider`` contract with one
extension: ``accepts_mask_set``. The job layer builds the semantic edit mask
(SCHP parsing + MediaPipe face guard) and hands the provider the whole
:class:`~app.domains.portrait.edit_mask.EditMaskSet`, because this graph needs
two *separate* masks -- the clothing pass and the hair pass run one after the
other -- plus the protected-identity mask the gate checks the crop against. A
single union mask cannot express that, so a run without the mask set is refused
rather than approximated.

What this module deliberately does not do:

* **No second job system.** It is one provider call; scheduling, status and
  downloads stay with the existing job layer.
* **No retry of a billed request.** ``create_task`` is issued exactly once. If
  its outcome is not known the provider raises and says so; it never resubmits.
* **No hard-coded style.** The prompts come from the plan the *user selected*;
  ``build_prompts`` is called with that plan's id as ``expected_style_id``, so a
  bundle can never be driven by a different look-alike style.
* **No local model, no GPU.** It is an HTTP client plus local mask bookkeeping.

Every gate check that guards the paid call (graph validation, denoise band,
stage records, mask geometry vs. the facial core, crop aspect) runs *before*
:func:`create_task`, so a graph that cannot be trusted never spends money.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from PIL import Image

from app.infrastructure import runninghub_crop_stitch as rh
from app.schemas.retouch import RetouchPlan

#: The three files the graph loads, under the names the local package uses.
STAGED_INPUTS = {
    "original": "original.jpg",
    "clothing_mask": "clothing_edit_mask.png",
    "hair_mask": "hair_edit_mask.png",
}

#: ``EditMaskSet`` attribute -> packaged file name for the three masks the graph
#: and the gate read. All three must be the source photo's own size.
PACKAGE_MASKS = {
    "hair_edit_mask.png": "hair",
    "clothing_edit_mask.png": "clothing",
    "protected_identity_mask.png": "protected",
}

#: Suffix of the clothing pass's own stitch, kept beside the final frame so the
#: UI can show which pass changed what.
CLOTHING_PASS_SUFFIX = "_clothing_pass"

METADATA_DIRNAME = "_meta"

#: Object tables are ~30 MB of JSON; load each configured file once per process.
_OBJECT_INFO_CACHE: dict[str, dict] = {}
_OBJECT_INFO_LOCK = threading.Lock()


class RunningHubCropStitchProviderError(RuntimeError):
    """A safe RunningHub error that never contains API credentials."""


def _refusal_message(exc: SystemExit) -> str:
    """Turn an infrastructure preflight refusal into an actionable user message.

    CropAndStitch only understands *styling* plans: the graph needs a hair or
    clothing target to rewrite. When the app hands it a plan without one -- most
    often the skin-retouch plan produced by the 皮肤质感精修 panel -- the prompt
    builder exits before any upload. That is a wrong-tool situation, not a
    provider outage, so say so plainly (and confirm nothing was billed) instead
    of echoing the raw marker wording. The refusal always happens before
    ``create_task``, so the "no charge" claim is guaranteed by the call order.
    """
    text = str(exc)
    if "target marker to extract" in text or "clause to build a prompt from" in text:
        return (
            "RunningHub 只支持「AI 造型设计」的造型方案（发型／服装）。"
            "当前提交的是皮肤美化方案，没有可改造的造型目标，已在自动上传前取消，未产生任何费用。"
            "请改用「本地模拟」图像服务商，或在上方改用「AI 造型设计」重新设计。"
        )
    if "expected" in text and "style_id" in text:
        return (
            "本次任务使用的造型方案与所选方案不一致，已拒绝提交以免生成错误风格，未产生任何费用。"
            "请重新选择造型方案后再生成。"
        )
    return f"RunningHub 拒绝了本次提交（未上传原图、未产生费用）：{text}"


def _load_object_info(path: Path | None) -> dict | None:
    """The live ComfyUI node table, or ``None`` for structural checks only."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    key = str(path)
    with _OBJECT_INFO_LOCK:
        cached = _OBJECT_INFO_CACHE.get(key)
        if cached is None:
            cached = json.loads(path.read_text(encoding="utf-8"))
            _OBJECT_INFO_CACHE[key] = cached
        return cached


class RunningHubCropStitchProvider:
    """``lquesada/ComfyUI-Inpaint-CropAndStitch`` two-pass edit over RunningHub."""

    provider_name = "runninghub-crop-stitch"
    output_extension = ".png"

    #: Cannot run without an explicit black/white edit mask: the job layer builds
    #: one instead of silently degrading to a whole-image edit.
    requires_edit_mask = True
    #: Wants the *set* (hair + clothing + protected), not just the union mask.
    accepts_mask_set = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = rh.BASE_URL,
        workflow_id: str = "1",
        instance_type: str = "default",
        seed: int = 20261008,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 6.0,
        checkpoint: str = rh.CHECKPOINT,
        filename_prefix: str = "p29_elegant",
        object_info_path: Path | None = None,
        key_file: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._workflow_id = workflow_id
        self._instance_type = instance_type
        self._seed = int(seed)
        self._timeout_seconds = float(timeout_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._checkpoint = checkpoint
        self._filename_prefix = filename_prefix
        self._object_info_path = Path(object_info_path) if object_info_path else None
        self._key_file = Path(key_file) if key_file else (Path.home() / ".runninghub_key")
        #: Public so the job layer can attach a per-job reporter after the
        #: provider is built by the factory.
        self.on_progress = on_progress
        self.model_name = f"RunningHub CropAndStitch · {checkpoint}"

        #: Filled in by :meth:`edit_image` for the job layer to record.
        self.last_task_id: str | None = None
        self.last_clothing_pass_path: Path | None = None
        self.last_gate_report_path: Path | None = None
        self.last_prompts: dict[str, Any] | None = None
        #: Billed RunningHub requests issued by this provider instance (0 or 1).
        self.paid_requests = 0

    # ------------------------------------------------------------------ public

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
        *,
        mask_path: Path | None = None,
        mask_set: Any | None = None,
        mask_build: Any | None = None,
    ) -> None:
        """Run the two-pass masked edit and write the raw RunningHub frame.

        The bytes written to ``output_path`` are the platform's own output,
        untouched: restoring the kept pixels is a separate, auditable step
        (``NativeMaskInpainting.finalise``) so the two can never be confused.
        """
        del mask_path, mask_build  # the mask *set* is what this graph needs
        await asyncio.to_thread(
            self._run_sync,
            Path(source_path),
            Path(output_path),
            plan,
            mask_set,
        )

    # --------------------------------------------------------------- transport

    def _notify(self, stage: str) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(stage)
            except Exception:  # pragma: no cover - progress must never fail a job
                pass

    def _resolve_key(self) -> str:
        """The RunningHub key, from the request/config or the credential file.

        Unlike the CLI's ``_read_key`` this raises a normal exception: a missing
        credential is a job failure, not a process exit, and a provider must never
        terminate the worker.
        """
        if len(self._api_key) >= 16:
            return self._api_key
        if self._key_file is not None and self._key_file.is_file():
            file_key = self._key_file.read_text(encoding="utf-8").strip()
            if len(file_key) >= 16:
                return file_key
        raise RunningHubCropStitchProviderError(
            "RunningHub API key is not configured. Set RUNNINGHUB_API_KEY or "
            "write the key to ~/.runninghub_key; no request was sent."
        )

    # ------------------------------------------------------------------- run

    def _run_sync(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        mask_set: Any | None,
    ) -> None:
        if not source_path.is_file():
            raise RunningHubCropStitchProviderError(
                f"Source image not found: {source_path.name}. No request was sent."
            )
        if mask_set is None:
            raise RunningHubCropStitchProviderError(
                "The RunningHub CropAndStitch backend needs the semantic mask set "
                "(hair + clothing + protected); a single union mask cannot run two "
                "passes. No request was sent."
            )
        style_id = str(getattr(plan, "plan_id", "") or "").strip()
        if not style_id:
            raise RunningHubCropStitchProviderError(
                "The selected plan has no id; refusing to guess which style to "
                "generate. No request was sent."
            )

        key = self._resolve_key()
        package = Path(tempfile.mkdtemp(prefix="rh-crop-stitch-"))
        try:
            # Preflight. Everything below this block must not reach the platform.
            self._notify("provider_preflight")
            bundle, prompts = self._build_prompts(plan, style_id)
            self.last_prompts = prompts
            frame = self._write_package(package, source_path, mask_set)
            workflow = rh.build_workflow(
                package,
                {kind: f"__UPLOAD__{kind}" for kind in STAGED_INPUTS},
                prompts,
                self._seed,
                filename_prefix=self._filename_prefix,
                checkpoint=self._checkpoint,
            )
            object_info = _load_object_info(self._object_info_path)
            errors = rh.validate_workflow(workflow, object_info)
            if errors:
                raise RunningHubCropStitchProviderError(
                    "RunningHub workflow failed validation; nothing was submitted: "
                    + "; ".join(errors)
                )
            gate_errors, gate = rh.collect_gate_checks(
                workflow,
                object_info,
                prompts,
                bundle,
                package,
                expected_style_id=style_id,
                checkpoint=self._checkpoint,
            )
            gate["validate_workflow_errors"] = errors
            gate["sourceFrame"] = {"width": frame[0], "height": frame[1]}
            gate["requestedBy"] = {
                "provider": self.provider_name,
                "checkpoint": self._checkpoint,
                "seed": self._seed,
                "workflowId": self._workflow_id,
                "instanceType": self._instance_type,
            }
            self.last_gate_report_path = self._write_audit(output_path, "gate", gate)
            if gate_errors:
                raise RunningHubCropStitchProviderError(
                    "RunningHub preflight gate rejected this run; no billed task was "
                    "created: " + "; ".join(gate_errors)
                )

            client = httpx.Client(base_url=self._base_url, timeout=60.0)
            try:
                self._notify("provider_upload")
                filenames = {
                    kind: rh.upload(client, key, package / name)
                    for kind, name in STAGED_INPUTS.items()
                }
                workflow = rh.build_workflow(
                    package,
                    filenames,
                    prompts,
                    self._seed,
                    filename_prefix=self._filename_prefix,
                    checkpoint=self._checkpoint,
                )
                errors = rh.validate_workflow(workflow, object_info)
                if errors:
                    raise RunningHubCropStitchProviderError(
                        "RunningHub workflow failed validation after upload; nothing "
                        "was submitted: " + "; ".join(errors)
                    )
                self._write_audit(output_path, "workflow", workflow)

                # The one and only billed call for this job.
                try:
                    created = rh.create_task(
                        client,
                        key,
                        workflow,
                        workflow_id=self._workflow_id,
                        instance_type=self._instance_type,
                    )
                except httpx.HTTPError as exc:
                    raise RunningHubCropStitchProviderError(
                        "The RunningHub create call did not return an answer, so it is "
                        "not known whether a task was billed. This run is NOT retried "
                        "automatically: check the RunningHub task list before "
                        f"submitting again. ({exc.__class__.__name__})"
                    ) from exc
                data = created.get("data") or {}
                task_id = str(data.get("taskId") or "")
                self.paid_requests = 1
                self.last_task_id = task_id
                self._write_audit(
                    output_path,
                    "task",
                    {"taskId": task_id, "create": created, "prompts": prompts},
                )
                self._notify("provider_submitted")

                status = rh.poll_task(
                    client,
                    key,
                    task_id,
                    timeout_seconds=self._timeout_seconds,
                    interval_seconds=self._poll_interval_seconds,
                    on_status=self._on_status,
                )
                if status != "SUCCESS":
                    raise rh.RunningHubError(
                        f"RunningHub task {task_id} did not succeed (status={status})"
                    )

                self._notify("provider_download")
                outputs = rh.fetch_outputs(client, key, task_id)
                images = [item for item in outputs if str(item.get("fileUrl") or "").strip()]
                if not images:
                    raise rh.RunningHubError(
                        f"RunningHub task {task_id} returned no image output"
                    )
                final = images[-1]
                clothing_pass = images[-2] if len(images) > 1 else None

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(rh.download(client, str(final["fileUrl"])))
                if clothing_pass is not None:
                    pass_path = output_path.with_name(
                        f"{output_path.stem}{CLOTHING_PASS_SUFFIX}{output_path.suffix}"
                    )
                    pass_path.write_bytes(rh.download(client, str(clothing_pass["fileUrl"])))
                    self.last_clothing_pass_path = pass_path
                self._write_meta(output_path, plan, prompts, task_id, outputs)
            finally:
                client.close()
        except SystemExit as exc:
            # ``build_prompts`` / ``_plan_target`` signal a refusal by exiting.
            # Inside a worker that would escape ``execute_job``'s ``except
            # Exception``, so it is converted here.
            raise RunningHubCropStitchProviderError(
                _refusal_message(exc)
            ) from exc
        finally:
            shutil.rmtree(package, ignore_errors=True)

    def _on_status(self, status: str) -> None:
        if status in {"SUCCESS", "FAILED"}:
            self._notify(f"provider_{status.lower()}")
        else:
            self._notify("provider_running")

    # ----------------------------------------------------------------- prompt

    def _build_prompts(self, plan: RetouchPlan, style_id: str) -> tuple[dict, dict]:
        """The plan the user picked, turned into the two pass prompts.

        The identity rule is synthesised rather than taken from the plan: the
        styling plans the UI submits carry no ``identity_constraints``, and the
        graph needs the one mask rule. ``build_prompts`` is asked to verify the
        bundle against the *selected* plan id, so a mismatch is refused.
        """
        edit_prompt = str(getattr(plan, "edit_prompt", "") or "").strip()
        if not edit_prompt:
            raise RunningHubCropStitchProviderError(
                "The selected plan has no edit prompt to build the RunningHub prompts "
                "from. No request was sent."
            )
        bundle = {
            "style_id": style_id,
            "style_name": str(getattr(plan, "title", "") or style_id),
            "hair_prompt": edit_prompt,
            "clothing_prompt": edit_prompt,
            "identity_constraints": {
                "rules": [{"id": rh.MASK_RULE_ID, "en": rh.DEFAULT_MASK_RULE}]
            },
        }
        prompts = rh.build_prompts(bundle, expected_style_id=style_id)
        return bundle, prompts

    # ---------------------------------------------------------------- package

    def _write_package(self, package: Path, source_path: Path, mask_set: Any) -> tuple[int, int]:
        """Stage the four same-size inputs and the manifest the gate reads."""
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            size = rgb.size
            rgb.save(package / "original.jpg", format="JPEG", quality=95)

        for name, attribute in PACKAGE_MASKS.items():
            array = getattr(mask_set, attribute, None)
            if array is None:
                raise RunningHubCropStitchProviderError(
                    f"The semantic mask set has no {attribute} mask; the RunningHub "
                    "graph cannot run without it. No request was sent."
                )
            values = np.asarray(array)
            if values.shape[:2] != (size[1], size[0]):
                raise RunningHubCropStitchProviderError(
                    f"The {attribute} mask is {values.shape[1]}x{values.shape[0]} but the "
                    f"photo is {size[0]}x{size[1]}; refusing to resize either. No request "
                    "was sent."
                )
            binary = np.where(values > 127, 255, 0).astype(np.uint8)
            Image.fromarray(binary, mode="L").save(package / name)

        (package / "manifest.json").write_text(
            json.dumps({"source_image": {"width": size[0], "height": size[1]}}),
            encoding="utf-8",
        )
        return size

    # ----------------------------------------------------------------- audit

    def _audit_dir(self, output_path: Path) -> Path:
        directory = output_path.parent / METADATA_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_audit(self, output_path: Path, kind: str, payload: Any) -> Path:
        target = self._audit_dir(output_path) / f"{output_path.name}.{kind}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _write_meta(
        self,
        output_path: Path,
        plan: RetouchPlan,
        prompts: dict,
        task_id: str,
        outputs: list[dict],
    ) -> None:
        """Non-secret provenance next to the generated image."""
        meta = {
            "provider": self.provider_name,
            "model": self.model_name,
            "mode": "runninghub-crop-stitch",
            "planId": plan.plan_id,
            "planTitle": getattr(plan, "title", None),
            "outputFile": output_path.name,
            "clothingPassFile": (
                self.last_clothing_pass_path.name if self.last_clothing_pass_path else None
            ),
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "taskId": task_id,
            "paidRequests": 1,
            "checkpoint": self._checkpoint,
            "seed": self._seed,
            "workflowId": self._workflow_id,
            "instanceType": self._instance_type,
            "inputs": sorted(STAGED_INPUTS),
            "prompts": {key: prompts.get(key) for key in ("clothing", "hair", "negative")},
            "outputs": [
                {"fileUrl": str(item.get("fileUrl") or ""), "consumeCoins": item.get("consumeCoins")}
                for item in outputs
            ],
            "styleIdMatchesPlan": prompts.get("style_id") == plan.plan_id,
            "note": "raw platform output; kept pixels are restored by inpaint_verify",
        }
        self._write_audit(output_path, "meta", meta)


__all__ = [
    "CLOTHING_PASS_SUFFIX",
    "METADATA_DIRNAME",
    "PACKAGE_MASKS",
    "STAGED_INPUTS",
    "RunningHubCropStitchProvider",
    "RunningHubCropStitchProviderError",
]
