from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.core.errors import bad_request, not_found
from app.providers.base import ImageEditProvider
from app.providers.mock_image_provider import MockImageProvider
from app.schemas.job import CreateRetouchJobRequest, RefineRetouchJobRequest, RetouchJob
from app.services.job_store import JobStore
from app.services.storage import StorageService


class RetouchService:
    def __init__(
        self,
        storage: StorageService,
        jobs: JobStore,
        provider: ImageEditProvider | None = None,
        brain: PromptBrain | None = None,
    ) -> None:
        self.storage = storage
        self.jobs = jobs
        self.provider = provider or MockImageProvider()
        self.brain = brain or LocalPromptBrain()

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
            output_image_id = f"out_{uuid4().hex[:12]}"
            output_filename = f"{output_image_id}{self.provider.output_extension}"
            output_path = self.storage.config.outputs_dir / output_filename
            await self.provider.edit_image(
                source_path=source_path,
                output_path=output_path,
                plan=effective_plan,
                user_instruction=job.user_instruction,
            )
            job.status = "succeeded"
            job.output_image_ids = [output_image_id]
            job.output_urls = [f"/data/outputs/{output_filename}"]
            job.updated_at = datetime.now(timezone.utc)
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
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
            output_image_id = f"out_{uuid4().hex[:12]}"
            output_filename = f"{output_image_id}{self.provider.output_extension}"
            output_path = self.storage.config.outputs_dir / output_filename
            await self.provider.edit_image(
                source_path=source_path,
                output_path=output_path,
                plan=effective_plan,
                user_instruction=request.user_instruction,
            )
            job.status = "succeeded"
            job.output_image_ids = [output_image_id]
            job.output_urls = [f"/data/outputs/{output_filename}"]
            job.updated_at = datetime.now(timezone.utc)
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)

        return self.jobs.save(job)

    def with_providers(
        self,
        provider: ImageEditProvider,
        brain: PromptBrain,
    ) -> "RetouchService":
        return RetouchService(self.storage, self.jobs, provider, brain)

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
