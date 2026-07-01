from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.domains.registry import domain_registry
from app.api.dependencies import (
    get_agent_brain_factory,
    get_provider_factory,
    get_retouch_service,
)
from app.brains.factory import AgentBrainFactory
from app.providers.factory import ImageProviderFactory
from app.schemas.job import CreateRetouchJobRequest, RefineRetouchJobRequest, RetouchJob
from app.schemas.retouch import RetouchPlan, RetouchPlansRequest
from app.services.retouch_service import RetouchService

router = APIRouter(prefix="/retouch", tags=["retouch"])


@router.post("/plans", response_model=list[RetouchPlan], response_model_by_alias=True)
def create_retouch_plans(request: RetouchPlansRequest) -> list[RetouchPlan]:
    planner = domain_registry.get_planner(request.analysis.domain_type)
    return planner.create_plans(request.analysis)


@router.get("/providers")
def get_retouch_providers(
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
) -> dict:
    return {
        **image_factory.capabilities(),
        **brain_factory.capabilities(),
    }


@router.post("/jobs", response_model=RetouchJob, response_model_by_alias=True)
def create_retouch_job(
    request: CreateRetouchJobRequest,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    provider_name: Annotated[str | None, Header(alias="X-AI-Provider")] = None,
    api_key: Annotated[str | None, Header(alias="X-AI-API-Key")] = None,
    workspace_id: Annotated[str | None, Header(alias="X-AI-Workspace-Id")] = None,
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
) -> RetouchJob:
    provider = image_factory.create(provider_name, api_key, workspace_id)
    brain = brain_factory.create(brain_name, brain_api_key)
    return service.with_providers(provider, brain).create_job(request)


@router.get("/jobs/{job_id}", response_model=RetouchJob, response_model_by_alias=True)
def get_retouch_job(
    job_id: str,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
) -> RetouchJob:
    return service.get_job(job_id)


@router.post("/jobs/{job_id}/refine", response_model=RetouchJob, response_model_by_alias=True)
def refine_retouch_job(
    job_id: str,
    request: RefineRetouchJobRequest,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    provider_name: Annotated[str | None, Header(alias="X-AI-Provider")] = None,
    api_key: Annotated[str | None, Header(alias="X-AI-API-Key")] = None,
    workspace_id: Annotated[str | None, Header(alias="X-AI-Workspace-Id")] = None,
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
) -> RetouchJob:
    provider = image_factory.create(provider_name, api_key, workspace_id)
    brain = brain_factory.create(brain_name, brain_api_key)
    return service.with_providers(provider, brain).refine_job(job_id, request)
