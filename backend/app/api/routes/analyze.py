from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_storage_service
from app.core.errors import not_found
from app.domains.registry import domain_registry
from app.schemas.analysis import AnalyzeRequest, PhotoAnalysis
from app.services.storage import StorageService

router = APIRouter(prefix="/photos", tags=["analysis"])


@router.post("/analyze", response_model=PhotoAnalysis, response_model_by_alias=True)
def analyze_photo(
    request: AnalyzeRequest,
    storage: Annotated[StorageService, Depends(get_storage_service)],
) -> PhotoAnalysis:
    image_path = storage.resolve_image_path(request.image_id)
    if image_path is None:
        raise not_found("Image not found.")

    analyzer = domain_registry.get_analyzer(request.domain_type)
    return analyzer.analyze(request.image_id, image_path)
