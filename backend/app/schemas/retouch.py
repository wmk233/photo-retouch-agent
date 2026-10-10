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

    # --- optional styling detail -------------------------------------------
    # The seven fields above are what the renderer needs. A styling proposal
    # carries more (the hair/makeup/outfit text and the design reasoning); those
    # are kept here so a submitted look is stored in full and the result page can
    # still describe what was generated. All optional, so the older plans that
    # only have the retouch fields keep validating.
    name: str | None = None
    design_axis: str | None = Field(default=None, alias="designAxis")
    design_direction: str | None = Field(default=None, alias="designDirection")
    design_reason: str | None = Field(default=None, alias="designReason")
    hair_plan: str | None = Field(default=None, alias="hairPlan")
    makeup_plan: str | None = Field(default=None, alias="makeupPlan")
    outfit_plan: str | None = Field(default=None, alias="outfitPlan")
    accessory_plan: str | None = Field(default=None, alias="accessoryPlan")
    palette: str | None = None
    identity_anchors: list[str] | None = Field(default=None, alias="identityAnchors")
    why_it_fits: str | None = Field(default=None, alias="whyItFits")
    estimated_paid_requests: int | None = Field(default=None, alias="estimatedPaidRequests")
    design_mode: str | None = Field(default=None, alias="designMode")
    source_provider: str | None = Field(default=None, alias="sourceProvider")
    selected_presets: list[str] | None = Field(default=None, alias="selectedPresets")
