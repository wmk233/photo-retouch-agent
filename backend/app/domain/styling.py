from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylingReference:
    """Where a proposal's styling vocabulary came from."""

    preset_id: str
    bucket: str
    name: str
    source_file: str
    vocabulary_ref: str
    prompt: str = ""

    def __post_init__(self) -> None:
        if not self.preset_id:
            raise ValueError("preset_id is required")


@dataclass(frozen=True)
class StylingProposal:
    plan_id: str
    domain_type: str
    title: str
    name: str
    design_axis: str
    design_direction: str
    description: str
    design_reason: str
    intensity: str
    hair_plan: str
    makeup_plan: str
    outfit_plan: str
    accessory_plan: str
    palette: str
    identity_anchors: tuple[str, ...] = ()
    why_it_fits: str = ""
    expected_changes: tuple[str, ...] = ()
    edit_prompt: str = ""
    negative_prompt: str = ""
    reference_presets: tuple[StylingReference, ...] = ()
    estimated_paid_requests: int = 1
    # Where the design came from. ``template`` is the free, deterministic
    # ``_Axis`` composition; ``ai`` is a real model-authored design. Keeping the
    # provenance on the proposal (instead of only in the UI) means a template
    # fallback can never be presented as a personalised AI design.
    design_mode: str = "template"
    source_provider: str = ""
    selected_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if not self.domain_type:
            raise ValueError("domain_type is required")
        if not self.design_axis:
            raise ValueError("design_axis is required")
        if self.design_mode not in {"template", "ai"}:
            raise ValueError("design_mode must be template or ai")
