from __future__ import annotations
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.retouch import RetouchPlan

RetouchJobStatus = Literal["queued", "running", "succeeded", "failed"]


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
    error_message: str | None = Field(default=None, alias="errorMessage")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
