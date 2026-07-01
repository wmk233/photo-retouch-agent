from app.schemas.retouch import RetouchPlan


class LocalPromptBrain:
    provider_name = "local"
    model_name = "rule-based-planner"

    def optimize(self, plan: RetouchPlan, user_instruction: str) -> RetouchPlan:
        if not user_instruction.strip():
            return plan
        return plan.model_copy(
            update={
                "edit_prompt": (
                    f"{plan.edit_prompt}\n用户补充要求：{user_instruction.strip()}"
                )
            }
        )
