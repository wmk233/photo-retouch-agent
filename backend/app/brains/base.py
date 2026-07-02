from pathlib import Path
from typing import Protocol

from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class PromptBrain(Protocol):
    provider_name: str
    model_name: str
    vision_mode: str

    def analyze(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        """Inspect a photo or enrich a deterministic baseline analysis."""

    def optimize(
        self,
        source_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> RetouchPlan:
        """Turn a structured plan and user instruction into an execution-ready prompt."""
