from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.schemas.photo import PhotoAsset
from app.services.storage import StorageService, storage_service

router = APIRouter(prefix="/photos", tags=["photos"])


def get_storage_service() -> StorageService:
    return storage_service


@router.post("/upload", response_model=PhotoAsset, response_model_by_alias=True)
async def upload_photo(
    file: Annotated[UploadFile, File()],
    service: Annotated[StorageService, Depends(get_storage_service)],
) -> PhotoAsset:
    return await service.save_upload(file)
