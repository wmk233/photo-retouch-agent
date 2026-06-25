from fastapi import APIRouter

from app.domains.registry import domain_registry
from app.schemas.retouch import RetouchPlan, RetouchPlansRequest

router = APIRouter(prefix="/retouch", tags=["retouch"])


@router.post("/plans", response_model=list[RetouchPlan], response_model_by_alias=True)
def create_retouch_plans(request: RetouchPlansRequest) -> list[RetouchPlan]:
    planner = domain_registry.get_planner(request.analysis.domain_type)
    return planner.create_plans(request.analysis)
