from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import rate_limit
from app.core.config import Settings
from app.main import create_app


def _app(tmp_path: Path, rate: int):
    settings = Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        brain_provider="local",
        face_lock_mode="off",
        rate_limit_per_second=rate,
    )
    return create_app(_settings=settings)


@pytest.fixture(autouse=True)
def _fresh_buckets():
    rate_limit._buckets.clear()
    yield
    rate_limit._buckets.clear()


def test_static_assets_do_not_consume_the_api_budget(tmp_path: Path) -> None:
    """A single page load is ~25 asset requests; it must never starve the API."""
    with TestClient(_app(tmp_path, rate=1)) as client:
        for _ in range(40):
            client.get("/")
        response = client.get("/api/health")

    assert response.status_code == 200


def test_api_over_limit_returns_429_not_500(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, rate=1)) as client:
        statuses = [client.get("/api/health").status_code for _ in range(5)]

    assert statuses[0] == 200
    assert 429 in statuses
    assert 500 not in statuses


def test_rate_limit_disabled_when_zero(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, rate=0)) as client:
        statuses = {client.get("/api/health").status_code for _ in range(30)}

    assert statuses == {200}
