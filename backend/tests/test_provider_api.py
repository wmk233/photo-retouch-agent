from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_brain_factory, get_provider_factory
from app.brains.factory import AgentBrainFactory
from app.brains.local import LocalPromptBrain
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


class CapturingBrainFactory(AgentBrainFactory):
    def __init__(self) -> None:
        self.received: dict[str, str | None] = {}

    def create(
        self,
        provider_name: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> LocalPromptBrain:
        self.received = {
            "provider": provider_name,
            "api_key": api_key,
            "workspace_id": workspace_id,
        }
        return LocalPromptBrain()

    def capabilities(self) -> dict:
        return {
            "defaultBrainProvider": "local",
            "deepseekConfigured": False,
            "deepseekModel": "deepseek-v4-flash",
            "glmConfigured": False,
            "glmModel": "glm-5v-turbo",
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
    payload = response.json()
    assert payload["defaultProvider"] == "mock"
    assert payload["defaultBrainProvider"] == "local"
    assert [item["id"] for item in payload["brainProviders"]] == [
        "qwen",
        "glm",
        "doubao",
        "deepseek",
        "local",
    ]
    assert [item["id"] for item in payload["actionProviders"]] == [
        "qwen",
        "wan",
        "seedream",
        "mock",
    ]
    assert payload["brainProviders"][0]["visionMode"] == "direct"
    assert next(
        item for item in payload["brainProviders"] if item["id"] == "deepseek"
    )["visionMode"] == "derived"
    assert "secret" not in response.text.lower()
    assert "apiKey" not in response.text


def test_request_api_key_is_not_persisted(
    client: TestClient,
    temp_settings,
) -> None:
    factory = CapturingProviderFactory()
    brain_factory = CapturingBrainFactory()
    client.app.dependency_overrides[get_provider_factory] = lambda: factory
    client.app.dependency_overrides[get_agent_brain_factory] = lambda: brain_factory
    image_id, plan = _create_plan(client)
    image_secret = "sk-image-secret-never-store"
    brain_secret = "sk-brain-secret-never-store"

    response = client.post(
        "/api/retouch/jobs",
        headers={
            "X-Action-Provider": "qwen",
            "X-Action-API-Key": image_secret,
            "X-Action-Workspace-Id": "workspace123",
            "X-Agent-Provider": "deepseek",
            "X-Agent-API-Key": brain_secret,
            "X-Agent-Workspace-Id": "brainspace",
        },
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 202
    assert factory.received == {
        "provider": "qwen",
        "api_key": image_secret,
        "workspace_id": "workspace123",
    }
    assert brain_factory.received == {
        "provider": "deepseek",
        "api_key": brain_secret,
        "workspace_id": "brainspace",
    }
    job_files = list(Path(temp_settings.jobs_dir).glob("*.json"))
    assert job_files
    assert all(
        image_secret not in path.read_text(encoding="utf-8")
        and brain_secret not in path.read_text(encoding="utf-8")
        for path in job_files
    )
