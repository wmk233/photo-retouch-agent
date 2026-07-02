from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.api.dependencies import (
    get_agent_brain_factory,
    get_provider_factory,
    get_storage_service,
)
from app.api.model_selection import require_model_selection
from app.brains.factory import AgentBrainFactory
from app.core.errors import not_found
from app.domains.registry import domain_registry
from app.providers.factory import ImageProviderFactory
from app.schemas.analysis import AnalyzeRequest, PhotoAnalysis
from app.services.storage import StorageService

router = APIRouter(prefix="/photos", tags=["analysis"])


@router.post("/analyze", response_model=PhotoAnalysis, response_model_by_alias=True)
def analyze_photo(
    request: AnalyzeRequest,
    storage: Annotated[StorageService, Depends(get_storage_service)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    action_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    brain_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
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
) -> PhotoAnalysis:
    image_path = storage.resolve_image_path(request.image_id)
    if image_path is None:
        raise not_found("Image not found.")

    analyzer = domain_registry.get_analyzer(request.domain_type)
    selected_brain = require_model_selection(
        brain_name,
        None,
        "Agent brain",
    )
    selected_action = require_model_selection(
        action_name,
        legacy_action_name,
        "Agent action",
    )
    brain = brain_factory.create(
        selected_brain,
        brain_api_key,
        brain_workspace_id,
    )
    action_factory.create(
        selected_action,
        action_api_key or legacy_action_api_key,
        action_workspace_id or legacy_action_workspace_id,
    )
    baseline = analyzer.analyze(request.image_id, image_path)
    return brain.analyze(request.image_id, image_path, baseline)
