from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.photos import get_storage_service
from app.core.config import Settings
from app.main import create_app
from app.services.storage import StorageService


@pytest.fixture()
def temp_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def client(temp_settings: Settings) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_storage_service] = lambda: StorageService(temp_settings)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_image_bytes(image_format: str = "PNG", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color=(36, 124, 114))
    image.save(buffer, format=image_format)
    return buffer.getvalue()
