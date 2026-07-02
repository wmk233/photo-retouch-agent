from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import (
    get_agent_brain_factory,
    get_job_store,
    get_provider_factory,
    get_storage_service,
)
from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.main import create_app
from app.providers.factory import ImageProviderFactory
from app.services.job_store import JobStore
from app.services.storage import StorageService


@pytest.fixture()
def temp_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        brain_provider="local",
        dashscope_api_key=None,
        dashscope_workspace_id=None,
        dashscope_endpoint=None,
        deepseek_api_key=None,
        zhipu_api_key=None,
        ark_api_key=None,
        rate_limit_per_second=0,
        max_concurrent_jobs=10,
    )


@pytest.fixture()
def client(temp_settings: Settings) -> Iterator[TestClient]:
    app = create_app(_settings=temp_settings)
    app.dependency_overrides[get_storage_service] = lambda: StorageService(temp_settings)
    app.dependency_overrides[get_job_store] = lambda: JobStore(temp_settings)
    app.dependency_overrides[get_provider_factory] = lambda: ImageProviderFactory(temp_settings)
    app.dependency_overrides[get_agent_brain_factory] = lambda: AgentBrainFactory(
        temp_settings
    )
    with TestClient(
        app,
        headers={
            "X-Agent-Provider": "local",
            "X-Action-Provider": "mock",
        },
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_image_bytes(image_format: str = "PNG", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color=(36, 124, 114))
    image.save(buffer, format=image_format)
    return buffer.getvalue()
