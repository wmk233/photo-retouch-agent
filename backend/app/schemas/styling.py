from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis import PhotoAnalysis


class StylingReference(BaseModel):
    """The stylekit preset a proposal was composed from (traceability)."""

    model_config = ConfigDict(populate_by_name=True)

    preset_id: str = Field(alias="presetId")
    bucket: str
    name: str
    source_file: str = Field(alias="sourceFile")
    vocabulary_ref: str = Field(alias="vocabularyRef")
    prompt: str = ""


class StylingProposal(BaseModel):
    """One complete look designed for the person in the uploaded photo.

    The retouch fields (``planId``/``editPrompt``/``negativePrompt``/... ) are
    intentionally identical to ``RetouchPlan`` so a chosen proposal can be sent
    straight to ``POST /api/retouch/jobs`` without any translation.
    """

    model_config = ConfigDict(populate_by_name=True)

    plan_id: str = Field(alias="planId")
    domain_type: str = Field(alias="domainType")
    title: str
    name: str
    design_axis: str = Field(alias="designAxis")
    design_direction: str = Field(alias="designDirection")
    description: str
    design_reason: str = Field(alias="designReason")
    intensity: str
    hair_plan: str = Field(alias="hairPlan")
    makeup_plan: str = Field(alias="makeupPlan")
    outfit_plan: str = Field(alias="outfitPlan")
    accessory_plan: str = Field(alias="accessoryPlan")
    palette: str
    identity_anchors: list[str] = Field(alias="identityAnchors")
    why_it_fits: str = Field(alias="whyItFits")
    expected_changes: list[str] = Field(alias="expectedChanges")
    edit_prompt: str = Field(alias="editPrompt")
    negative_prompt: str = Field(alias="negativePrompt")
    reference_presets: list[StylingReference] = Field(
        default_factory=list, alias="referencePresets"
    )
    estimated_paid_requests: int = Field(default=1, alias="estimatedPaidRequests")
    # Provenance: ``template`` is the free deterministic composition, ``ai`` is a
    # design authored by a real model. Surfaced so the UI can never present a
    # template fallback as a personalised AI design.
    design_mode: str = Field(default="template", alias="designMode")
    source_provider: str = Field(default="", alias="sourceProvider")
    selected_presets: list[str] = Field(
        default_factory=list, alias="selectedPresets"
    )


class StylingProposalsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis: PhotoAnalysis
    # ``template`` keeps the free, zero-cost behaviour. ``ai`` asks the styling
    # planner to author three looks with a real vision model (one paid request).
    mode: str = "template"
    # Optional free-text wishes the person typed before designing (e.g. "想剪短").
    # Passed to the planner once; it never triggers an extra request.
    user_instruction: str = Field(default="", alias="userInstruction")
