from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.api.dependencies import (
    get_agent_brain_factory,
    get_provider_factory,
    get_retouch_service,
)
from app.api.model_selection import require_model_selection
from app.brains.base import PromptBrain
from app.brains.factory import AgentBrainFactory
from app.domains.registry import domain_registry
from app.providers.base import ImageEditProvider
from app.providers.factory import ImageProviderFactory
from app.schemas.job import CreateRetouchJobRequest, RefineRetouchJobRequest, RetouchJob
from app.schemas.retouch import RetouchPlan, RetouchPlansRequest
from app.services.retouch_service import RetouchService

router = APIRouter(prefix="/retouch", tags=["retouch"])


def _create_workflow_models(
    image_factory: ImageProviderFactory,
    brain_factory: AgentBrainFactory,
    action_name: str | None,
    action_api_key: str | None,
    action_workspace_id: str | None,
    legacy_action_name: str | None,
    legacy_action_api_key: str | None,
    legacy_action_workspace_id: str | None,
    brain_name: str | None,
    brain_api_key: str | None,
    brain_workspace_id: str | None,
) -> tuple[ImageEditProvider, PromptBrain]:
    selected_action = require_model_selection(
        action_name,
        legacy_action_name,
        "Agent action",
    )
    selected_brain = require_model_selection(
        brain_name,
        None,
        "Agent brain",
    )
    provider = image_factory.create(
        selected_action,
        action_api_key or legacy_action_api_key,
        action_workspace_id or legacy_action_workspace_id,
    )
    brain = brain_factory.create(
        selected_brain,
        brain_api_key,
        brain_workspace_id,
    )
    return provider, brain


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
async def create_retouch_job(
    request: CreateRetouchJobRequest,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    action_name: Annotated[str | None, Header(alias="X-Action-Provider")] = None,
    action_api_key: Annotated[str | None, Header(alias="X-Action-API-Key")] = None,
    action_workspace_id: Annotated[
        str | None, Header(alias="X-Action-Workspace-Id")
    ] = None,
    legacy_action_name: Annotated[str | None, Header(alias="X-AI-Provider")] = None,
    legacy_action_api_key: Annotated[
        str | None, Header(alias="X-AI-API-Key")
    ] = None,
    legacy_action_workspace_id: Annotated[
        str | None, Header(alias="X-AI-Workspace-Id")
    ] = None,
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    brain_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
) -> RetouchJob:
    provider, brain = _create_workflow_models(
        image_factory,
        brain_factory,
        action_name,
        action_api_key,
        action_workspace_id,
        legacy_action_name,
        legacy_action_api_key,
        legacy_action_workspace_id,
        brain_name,
        brain_api_key,
        brain_workspace_id,
    )
    return await service.with_providers(provider, brain).create_job(request)


@router.get("/jobs/{job_id}", response_model=RetouchJob, response_model_by_alias=True)
def get_retouch_job(
    job_id: str,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
) -> RetouchJob:
    return service.get_job(job_id)


@router.post("/jobs/{job_id}/refine", response_model=RetouchJob, response_model_by_alias=True)
async def refine_retouch_job(
    job_id: str,
    request: RefineRetouchJobRequest,
    service: Annotated[RetouchService, Depends(get_retouch_service)],
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    action_name: Annotated[str | None, Header(alias="X-Action-Provider")] = None,
    action_api_key: Annotated[str | None, Header(alias="X-Action-API-Key")] = None,
    action_workspace_id: Annotated[
        str | None, Header(alias="X-Action-Workspace-Id")
    ] = None,
    legacy_action_name: Annotated[str | None, Header(alias="X-AI-Provider")] = None,
    legacy_action_api_key: Annotated[
        str | None, Header(alias="X-AI-API-Key")
    ] = None,
    legacy_action_workspace_id: Annotated[
        str | None, Header(alias="X-AI-Workspace-Id")
    ] = None,
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    brain_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
) -> RetouchJob:
    provider, brain = _create_workflow_models(
        image_factory,
        brain_factory,
        action_name,
        action_api_key,
        action_workspace_id,
        legacy_action_name,
        legacy_action_api_key,
        legacy_action_workspace_id,
        brain_name,
        brain_api_key,
        brain_workspace_id,
    )
    return await service.with_providers(provider, brain).refine_job(job_id, request)
