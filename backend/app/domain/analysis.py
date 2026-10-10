from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectSummary:
    count: int
    position: str
    face_visibility: str

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("subject count cannot be negative")


@dataclass(frozen=True)
class PhotoAnalysis:
    image_id: str
    domain_type: str
    scene_type: str
    subjects: SubjectSummary
    lighting_issues: tuple[str, ...] = ()
    background_issues: tuple[str, ...] = ()
    portrait_suggestions: tuple[str, ...] = ()
    composition_suggestions: tuple[str, ...] = ()
    recommended_styles: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    brain_provider: str = "local"
    brain_model: str = "rule-based-analyzer"
    vision_mode: str = "derived"
    face_facts: dict | None = None
    #: True when the analysis was served from the reuse cache, i.e. this call
    #: did not reach the model provider and therefore cost nothing.
    analysis_reused: bool = False

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id is required")
        if not self.domain_type:
            raise ValueError("domain_type is required")
