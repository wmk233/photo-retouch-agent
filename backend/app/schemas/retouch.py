from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis import PhotoAnalysis


class RetouchPlansRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis: PhotoAnalysis


class RetouchPlan(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    plan_id: str = Field(alias="planId")
    domain_type: str = Field(alias="domainType")
    title: str
    description: str
    intensity: str
    edit_prompt: str = Field(alias="editPrompt")
    negative_prompt: str = Field(alias="negativePrompt")
    expected_changes: list[str] = Field(alias="expectedChanges")
