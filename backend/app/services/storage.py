from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, settings
from app.core.errors import bad_request, payload_too_large
from app.schemas.photo import PhotoAsset


CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def ensure_data_dirs(config: Settings = settings) -> None:
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.jobs_dir.mkdir(parents=True, exist_ok=True)


class StorageService:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        ensure_data_dirs(config)

    async def save_upload(self, upload: UploadFile) -> PhotoAsset:
        content_type = upload.content_type or ""
        if content_type not in self.config.allowed_image_types:
            raise bad_request("Only JPG, PNG, and WebP images are supported.")

        content = await upload.read()
        size_bytes = len(content)
        if size_bytes > self.config.max_upload_bytes:
            raise payload_too_large("Image must be 10MB or smaller.")

        width, height = self._read_dimensions(content)
        image_id = f"img_{uuid4().hex[:12]}"
        filename = f"{image_id}{CONTENT_TYPE_EXTENSIONS[content_type]}"
        path = self.config.uploads_dir / filename
        path.write_bytes(content)

        return PhotoAsset(
            image_id=image_id,
            filename=filename,
            content_type=content_type,
            width=width,
            height=height,
            size_bytes=size_bytes,
            url=f"/data/uploads/{filename}",
            created_at=datetime.now(timezone.utc),
        )

    def resolve_image_path(self, image_id: str) -> Path | None:
        matches = list(self.config.uploads_dir.glob(f"{image_id}.*"))
        if matches:
            return matches[0]
        matches = list(self.config.outputs_dir.glob(f"{image_id}.*"))
        if matches:
            return matches[0]
        return None

    @staticmethod
    def _read_dimensions(content: bytes) -> tuple[int, int]:
        try:
            with Image.open(BytesIO(content)) as image:
                return image.size
        except UnidentifiedImageError as exc:
            raise bad_request("Uploaded file is not a readable image.") from exc


storage_service = StorageService()
