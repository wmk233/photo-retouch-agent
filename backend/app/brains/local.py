from __future__ import annotations

from pathlib import Path

from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class LocalPromptBrain:
    provider_name = "local"
    model_name = "rule-based-planner"
    vision_mode = "derived"

    async def analyze(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        return baseline.model_copy(
            update={
                "brain_provider": self.provider_name,
                "brain_model": self.model_name,
                "vision_mode": self.vision_mode,
            }
        )

    async def optimize(
        self,
        source_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> RetouchPlan:
        # The supplementary instruction is not folded into ``edit_prompt`` here.
        # Every image provider already appends it once, as the last hop before
        # the request is sent (``_build_prompt``), so adding it again would put
        # the same sentence into the model request twice.
        return plan
