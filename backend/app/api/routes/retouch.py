from typing import Annotated

from fastapi import APIRouter, Depends

from app.domains.registry import domain_registry
from app.api.dependencies import get_retouch_service
from app.schemas.job import CreateRetouchJobRequest, RetouchJob
from app.schemas.retouch import RetouchPlan, RetouchPlansRequest
from app.services.retouch_service import RetouchService

router = APIRouter(prefix="/retouch", tags=["retouch"])


@router.post("/plans", response_model=list[RetouchPlan], response_model_by_alias=True)
def create_retouch_plans(request: RetouchPlansRequest) -> list[RetouchPlan]:
    planner = domain_registry.get_planner(request.analysis.domain_type)
    return planner.create_plans(request.analysis)


@router.post("/jobs", response_model=RetouchJob, response_model_by_alias=True)
def create_retouch_job(
    request: CreateRetouchJobRequest,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
) -> RetouchJob:
    return service.create_job(request)


@router.get("/jobs/{job_id}", response_model=RetouchJob, response_model_by_alias=True)
def get_retouch_job(
    job_id: str,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
) -> RetouchJob:
    return service.get_job(job_id)
