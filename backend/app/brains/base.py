from typing import Protocol

from app.schemas.retouch import RetouchPlan


class PromptBrain(Protocol):
    provider_name: str
    model_name: str

    def optimize(self, plan: RetouchPlan, user_instruction: str) -> RetouchPlan:
        """Turn a structured plan and user instruction into an execution-ready prompt."""
