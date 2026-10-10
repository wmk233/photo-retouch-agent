from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.core.config import Settings, settings as default_settings
from app.core.errors import bad_request, not_found
from app.domains.portrait import face_lock as face_lock_domain
from app.domains.portrait.inpaint_verify import InpaintVerification
from app.providers.base import ImageEditProvider
from app.providers.mock_image_provider import MockImageProvider
from app.schemas.job import CreateRetouchJobRequest, RefineRetouchJobRequest, RetouchJob
from app.services.job_store import JobStore
from app.services.native_mask_inpainting import NativeMaskInpainting
from app.services.storage import StorageService


class RetouchService:
    def __init__(
        self,
        storage: StorageService,
        jobs: JobStore,
        provider: ImageEditProvider | None = None,
        brain: PromptBrain | None = None,
        settings: Settings | None = None,
        native_mask: NativeMaskInpainting | None = None,
    ) -> None:
        self.storage = storage
        self.jobs = jobs
        self.provider = provider or MockImageProvider()
        self.brain = brain or LocalPromptBrain()
        self.settings = settings or default_settings
        self.native_mask = native_mask or NativeMaskInpainting(self.settings)

    # ------------------------------------------------------------- face lock
    def _run_face_lock(
        self,
        source_path,
        generated_path,
        generated_image_id: str,
    ) -> face_lock_domain.FaceLockReport | None:
        """Restore the original face into the generated image (strict mode only).

        Returns ``None`` when the lock is switched off, so the caller keeps the
        previous behaviour untouched. A returned report is *never* silently
        treated as success: the caller checks :attr:`FaceLockReport.ok` and fails
        the job when the strict lock could not be verified.
        """
        config = face_lock_domain.config_from_settings(self.settings)
        if not getattr(self.settings, "face_lock_enabled", False):
            return None
        if config.mode != face_lock_domain.FACE_LOCK_STRICT:
            return None

        locked_image_id = f"lock_{uuid4().hex[:12]}"
        locked_filename = f"{locked_image_id}.png"
        locked_path = self.storage.config.outputs_dir / locked_filename

        report = face_lock_domain.apply_face_lock(
            original_path=source_path,
            generated_path=generated_path,
            output_path=locked_path,
            model_path=getattr(self.settings, "face_landmark_model_path", None),
            config=config,
            manual_mask_path=getattr(self.settings, "face_lock_manual_mask_path", None),
        )
        report.diagnostics["preLockImageId"] = generated_image_id
        report.diagnostics["preLockPath"] = f"/data/outputs/{generated_path.name}"
        report.diagnostics["lockedImageId"] = locked_image_id if report.ok else None
        report.diagnostics["lockedPath"] = (
            f"/data/outputs/{locked_filename}" if report.ok else None
        )
        report.diagnostics["manualMaskPath"] = (
            str(self.settings.face_lock_manual_mask_path)
            if getattr(self.settings, "face_lock_manual_mask_path", None)
            else None
        )
        return report

    def _finish_job(
        self,
        job: RetouchJob,
        source_path,
        output_image_id: str,
        output_filename: str,
        output_path,
    ) -> RetouchJob:
        """Record the provider output, applying the strict face lock when enabled."""
        report = self._run_face_lock(source_path, output_path, output_image_id)
        if report is not None:
            job.face_lock = report.to_dict()
        if report is None or report.ok:
            job.status = "succeeded"
            job.verification_level = "visual_pending"
            job.stage = "done"
            if report is None:
                job.output_image_ids = [output_image_id]
                job.output_urls = [f"/data/outputs/{output_filename}"]
            else:
                locked_filename = f"{report.diagnostics['lockedImageId']}.png"
                job.output_image_ids = [report.diagnostics["lockedImageId"]]
                job.output_urls = [f"/data/outputs/{locked_filename}"]
                job.error_message = None
        else:
            # A styling render whose strict face lock failed is not a success.
            job.status = "failed"
            job.verification_level = "visual_failed"
            job.stage = "done"
            job.output_image_ids = []
            job.output_urls = []
            job.error_message = (
                f"保脸失败（{report.status.value}）：{report.message} "
                "生成图未通过严格人脸锁定校验，请人工检查保护区域或调整掩膜后重试。"
            )
        job.updated_at = datetime.now(timezone.utc)
        return job

    # ------------------------------------------------------ native mask flow
    def _is_native_mask_provider(self) -> bool:
        """True when the active backend cannot run without a black/white mask.

        The SiliconFlow and mock backends edit the whole image and never see a
        mask; the fal.ai inpainting backend declares ``requires_edit_mask`` and
        then *requires* one. There is no fallback between the two: a masked run
        whose mask cannot be built fails instead of silently degrading to a
        whole-image edit that would repaint the face.
        """
        return bool(getattr(self.provider, "requires_edit_mask", False))

    async def _render(self, job: RetouchJob, source_path, plan) -> None:
        """Run the single provider call for this job and record the outcome."""
        output_image_id = f"out_{uuid4().hex[:12]}"
        output_filename = f"{output_image_id}{self.provider.output_extension}"
        output_path = self.storage.config.outputs_dir / output_filename
        if self._is_native_mask_provider():
            await self._render_native_mask(
                job,
                source_path,
                plan,
                output_image_id,
                output_filename,
                output_path,
            )
        else:
            self._set_stage(job, "provider_running")
            await self.provider.edit_image(
                source_path=source_path,
                output_path=output_path,
                plan=plan,
                user_instruction=job.user_instruction,
            )
            self._finish_job(
                job, source_path, output_image_id, output_filename, output_path
            )

    def _set_stage(self, job: RetouchJob, stage: str) -> None:
        """Record the current step (best effort: a write failure must not fail a job)."""
        job.stage = stage
        job.updated_at = datetime.now(timezone.utc)
        try:
            self.jobs.save(job)
        except Exception:  # pragma: no cover - progress is advisory
            pass

    async def _render_native_mask(
        self,
        job: RetouchJob,
        source_path,
        plan,
        output_image_id: str,
        output_filename: str,
        output_path,
    ) -> None:
        """Build the mask locally, edit only through it, then keep the pixels."""
        mask_filename = f"mask_{uuid4().hex[:12]}.png"
        mask_path = self.storage.config.outputs_dir / mask_filename
        record: dict = {
            "backend": self.provider.provider_name,
            "model": self.provider.model_name,
            "maskPath": f"/data/outputs/{mask_filename}",
        }
        # Record the intent up front so a failed build still leaves a trace of
        # which mask file the run was supposed to use.
        job.native_mask = record
        job.verification_level = None
        self._set_stage(job, "mask_build")

        build = self.native_mask.build_mask(source_path, mask_path)
        record["mask"] = build.to_dict()

        # The provider writes its own raw output; the kept pixels are only
        # composited back by ``finalise`` once the raw output verifies.
        raw_filename = f"raw_{output_image_id}{self.provider.output_extension}"
        raw_path = self.storage.config.outputs_dir / raw_filename
        edit_kwargs: dict = {"mask_path": mask_path}
        if getattr(self.provider, "accepts_mask_set", False):
            edit_kwargs["mask_set"] = build.mask_set
            edit_kwargs["mask_build"] = build
            # The provider reports its own stage transitions from the worker
            # thread; attach the reporter for this job only.
            if hasattr(self.provider, "on_progress"):
                self.provider.on_progress = lambda stage: self._set_stage(job, stage)
        self._set_stage(job, "provider_preflight")
        await self.provider.edit_image(
            source_path=source_path,
            output_path=raw_path,
            plan=plan,
            user_instruction=job.user_instruction,
            **edit_kwargs,
        )
        record["rawPath"] = f"/data/outputs/{raw_filename}"
        provider_task_id = getattr(self.provider, "last_task_id", None)
        if provider_task_id:
            record["providerTaskId"] = provider_task_id
            job.provider_task_id = str(provider_task_id)
        paid = int(getattr(self.provider, "paid_requests", 0) or 0)
        if paid:
            job.paid_requests = max(job.paid_requests, paid)
            record["paidRequests"] = job.paid_requests
        clothing_pass = getattr(self.provider, "last_clothing_pass_path", None)
        if clothing_pass:
            record["clothingPassPath"] = f"/data/outputs/{Path(clothing_pass).name}"

        self._set_stage(job, "verification")
        verification = self.native_mask.finalise(
            source_path, raw_path, mask_path, output_path, mask_set=build.mask_set
        )
        record["verification"] = verification.to_dict()
        job.native_mask = record

        if verification.ok:
            job.status = "succeeded"
            # Machine checks passed; a human still has to accept the repainted
            # region, so the published result is never called visually verified.
            job.verification_level = "visual_pending"
            job.stage = "done"
            job.output_image_ids = [output_image_id]
            job.output_urls = [f"/data/outputs/{output_filename}"]
            job.error_message = None
        else:
            # The kept pixels did not prove themselves the original's: the raw
            # repaint is *not* published and no success status is written.
            job.status = "failed"
            job.verification_level = "visual_failed"
            job.stage = "done"
            job.output_image_ids = []
            job.output_urls = []
            job.error_message = (
                "原生蒙版改造未通过像素身份校验"
                "（PIXEL_IDENTITY_PASSED=false）："
                + "；".join(verification.reasons or ["输出图与原图无法可靠配准"])
                + "。已拒绝写入成功状态，请人工复核蒙版或改用整图后端。"
            )
        job.updated_at = datetime.now(timezone.utc)

    async def create_job_queued(self, request: CreateRetouchJobRequest) -> RetouchJob:
        """Create a job in 'queued' status without executing the AI workflow."""
        base_image_id = request.base_image_id or request.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)
        if source_path is None:
            raise not_found("Base image not found.")

        now = datetime.now(timezone.utc)
        job = RetouchJob(
            job_id=f"job_{uuid4().hex[:12]}",
            source_image_id=request.source_image_id,
            base_image_id=request.base_image_id,
            plan_id=request.plan.plan_id,
            plan=request.plan,
            user_instruction=request.user_instruction,
            model_provider=self.provider.provider_name,
            model_name=self.provider.model_name,
            brain_provider=self.brain.provider_name,
            brain_model=self.brain.model_name,
            status="queued",
            output_image_ids=[],
            output_urls=[],
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs.save(job)
        return job

    async def execute_job(self, job_id: str) -> RetouchJob:
        """Execute the AI workflow for a queued job."""
        job = self.jobs.get(job_id)
        base_image_id = job.base_image_id or job.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        self.jobs.save(job)

        try:
            effective_plan = await self.brain.optimize(
                source_path,
                job.plan,
                job.user_instruction,
            )
            job.plan = effective_plan
            await self._render(job, source_path, effective_plan)
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
            job.stage = "done"
            job.verification_level = "visual_failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)

        return self.jobs.save(job)

    async def create_job(self, request: CreateRetouchJobRequest) -> RetouchJob:
        base_image_id = request.base_image_id or request.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)
        if source_path is None:
            raise not_found("Base image not found.")

        now = datetime.now(timezone.utc)
        job = RetouchJob(
            job_id=f"job_{uuid4().hex[:12]}",
            source_image_id=request.source_image_id,
            base_image_id=request.base_image_id,
            plan_id=request.plan.plan_id,
            plan=request.plan,
            user_instruction=request.user_instruction,
            model_provider=self.provider.provider_name,
            model_name=self.provider.model_name,
            brain_provider=self.brain.provider_name,
            brain_model=self.brain.model_name,
            status="running",
            output_image_ids=[],
            output_urls=[],
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs.save(job)

        try:
            effective_plan = await self.brain.optimize(
                source_path,
                request.plan,
                request.user_instruction,
            )
            job.plan = effective_plan
            await self._render(job, source_path, effective_plan)
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
            job.stage = "done"
            job.verification_level = "visual_failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)

        return self.jobs.save(job)

    def with_providers(
        self,
        provider: ImageEditProvider,
        brain: PromptBrain,
    ) -> "RetouchService":
        return RetouchService(
            self.storage,
            self.jobs,
            provider,
            brain,
            self.settings,
            native_mask=self.native_mask,
        )

    def get_job(self, job_id: str) -> RetouchJob:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[RetouchJob]:
        return self.jobs.list_jobs(limit=limit, offset=offset)

    async def refine_job(self, job_id: str, request: RefineRetouchJobRequest) -> RetouchJob:
        parent = self.jobs.get(job_id)
        if parent.status != "succeeded" or not parent.output_image_ids:
            raise bad_request("Only succeeded jobs with output can be refined.")

        plan = request.plan or parent.plan
        if plan is None:
            raise bad_request("A retouch plan is required for refinement.")

        return await self.create_job(
            CreateRetouchJobRequest(
                source_image_id=parent.source_image_id,
                base_image_id=parent.output_image_ids[-1],
                plan=plan,
                user_instruction=request.user_instruction,
            )
        )
