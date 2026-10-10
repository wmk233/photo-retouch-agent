from __future__ import annotations
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.retouch import RetouchPlan

RetouchJobStatus = Literal["queued", "running", "succeeded", "failed"]

#: How far the render has progressed, so the UI can name the current step instead
#: of only "running". ``None`` for jobs created before this field existed.
RetouchJobStage = Literal[
    "queued",
    "segmentation",
    "mask_build",
    "provider_preflight",
    "provider_upload",
    "provider_submitted",
    "provider_running",
    "provider_download",
    "verification",
    "done",
]

#: The three independent verdicts a masked render can end on.
#: ``technical_pass`` -- machine checks passed and the effect is published.
#: ``visual_pending`` -- machine checks passed but a human still has to accept it.
#: ``visual_failed`` -- a hard error (identity broken, extra face, size mismatch).
RetouchJobVerificationLevel = Literal["technical_pass", "visual_pending", "visual_failed"]


class CreateRetouchJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_image_id: str = Field(alias="sourceImageId")
    base_image_id: str | None = Field(default=None, alias="baseImageId")
    plan: RetouchPlan
    user_instruction: str = Field(default="", alias="userInstruction")


class RefineRetouchJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_instruction: str = Field(alias="userInstruction")
    plan: RetouchPlan | None = None


class RetouchJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    source_image_id: str = Field(alias="sourceImageId")
    base_image_id: str | None = Field(default=None, alias="baseImageId")
    plan_id: str | None = Field(default=None, alias="planId")
    plan: RetouchPlan | None = None
    user_instruction: str = Field(default="", alias="userInstruction")
    model_provider: str = Field(alias="modelProvider")
    model_name: str = Field(alias="modelName")
    brain_provider: str = Field(default="local", alias="brainProvider")
    brain_model: str = Field(default="rule-based-planner", alias="brainModel")
    status: RetouchJobStatus
    output_image_ids: list[str] = Field(default_factory=list, alias="outputImageIds")
    output_urls: list[str] = Field(default_factory=list, alias="outputUrls")
    face_lock: dict | None = Field(default=None, alias="faceLock")
    native_mask: dict | None = Field(default=None, alias="nativeMask")
    stage: str | None = Field(default=None, alias="stage")
    paid_requests: int = Field(default=0, alias="paidRequests")
    provider_task_id: str | None = Field(default=None, alias="providerTaskId")
    verification_level: str | None = Field(default=None, alias="verificationLevel")
    error_message: str | None = Field(default=None, alias="errorMessage")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
