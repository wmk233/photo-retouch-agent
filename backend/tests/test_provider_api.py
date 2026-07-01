from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_provider_factory
from app.providers.factory import ImageProviderFactory
from app.providers.mock_image_provider import MockImageProvider
from tests.conftest import make_image_bytes


class CapturingProviderFactory(ImageProviderFactory):
    def __init__(self) -> None:
        self.received: dict[str, str | None] = {}

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> MockImageProvider:
        self.received = {
            "provider": provider_name,
            "api_key": api_key,
            "workspace_id": workspace_id,
        }
        return MockImageProvider()

    def capabilities(self) -> dict:
        return {
            "defaultProvider": "mock",
            "qwenConfigured": False,
            "qwenModel": "qwen-image-2.0-pro",
            "workspaceConfigured": False,
        }


def _create_plan(client: TestClient) -> tuple[str, dict]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_image_bytes(size=(640, 960)), "image/png")},
    )
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    return image_id, plans.json()[0]


def test_provider_capabilities_are_safe(client: TestClient) -> None:
    response = client.get("/api/retouch/providers")

    assert response.status_code == 200
    assert response.json() == {
        "defaultProvider": "mock",
        "qwenConfigured": False,
        "qwenModel": "qwen-image-2.0-pro",
        "workspaceConfigured": False,
    }
    assert "api" not in response.text.lower()


def test_request_api_key_is_not_persisted(
    client: TestClient,
    temp_settings,
) -> None:
    factory = CapturingProviderFactory()
    client.app.dependency_overrides[get_provider_factory] = lambda: factory
    image_id, plan = _create_plan(client)
    secret = "sk-user-secret-never-store"

    response = client.post(
        "/api/retouch/jobs",
        headers={
            "X-AI-Provider": "qwen",
            "X-AI-API-Key": secret,
            "X-AI-Workspace-Id": "workspace123",
        },
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 200
    assert factory.received == {
        "provider": "qwen",
        "api_key": secret,
        "workspace_id": "workspace123",
    }
    job_files = list(Path(temp_settings.jobs_dir).glob("*.json"))
    assert job_files
    assert all(secret not in path.read_text(encoding="utf-8") for path in job_files)
