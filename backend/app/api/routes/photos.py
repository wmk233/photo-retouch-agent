from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_storage_service
from app.schemas.photo import PhotoAsset
from app.services.storage import StorageService

router = APIRouter(prefix="/photos", tags=["photos"])


@router.post("/upload", response_model=PhotoAsset, response_model_by_alias=True)
async def upload_photo(
    file: Annotated[UploadFile, File()],
    service: Annotated[StorageService, Depends(get_storage_service)],
) -> PhotoAsset:
    return await service.save_upload(file)
