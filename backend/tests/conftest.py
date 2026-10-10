from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

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
        siliconflow_api_key=None,
        # Existing suites exercise the provider/job plumbing with synthetic
        # imagery that has no real face; the strict lock is covered separately
        # in tests/test_face_lock.py with an injected stub detector.
        face_lock_mode="off",
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


def make_portrait_bytes(size: tuple[int, int] = (640, 960)) -> bytes:
    """A synthetic vertical frame whose centre is filled with a skin-tone oval."""
    image = Image.new("RGB", size, color=(198, 205, 212))
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.ellipse(
        (width * 0.18, height * 0.10, width * 0.82, height * 0.62),
        fill=(221, 168, 138),
    )
    draw.rectangle(
        (width * 0.10, height * 0.62, width * 0.90, height * 1.0),
        fill=(58, 62, 74),
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_landscape_bytes(size: tuple[int, int] = (960, 640)) -> bytes:
    """A synthetic frame with sky and grass only - no skin-tone subject."""
    image = Image.new("RGB", size, color=(80, 130, 190))
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle((0, height * 0.55, width, height), fill=(90, 150, 80))
    draw.polygon(
        [(0, height * 0.75), (width * 0.5, height * 0.2), (width, height * 0.75)],
        fill=(120, 132, 128),
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
