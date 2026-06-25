from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from tests.conftest import make_image_bytes


def _analysis_and_plan(client: TestClient) -> tuple[str, dict]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_image_bytes(size=(640, 960)), "image/png")},
    )
    assert upload.status_code == 200
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    assert analysis.status_code == 200
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    assert plans.status_code == 200
    return image_id, plans.json()[0]


def test_create_retouch_job_generates_output(
    client: TestClient,
    temp_settings: Settings,
) -> None:
    image_id, plan = _analysis_and_plan(client)

    response = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 200
    job = response.json()
    assert job["jobId"].startswith("job_")
    assert job["sourceImageId"] == image_id
    assert job["planId"] == "natural"
    assert job["status"] == "succeeded"
    assert len(job["outputImageIds"]) == 1
    assert len(job["outputUrls"]) == 1
    assert Path(temp_settings.outputs_dir / f"{job['outputImageIds'][0]}.jpg").exists()


def test_get_retouch_job_returns_saved_job(client: TestClient) -> None:
    image_id, plan = _analysis_and_plan(client)
    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": "脸部再亮一点"},
    ).json()

    response = client.get(f"/api/retouch/jobs/{created['jobId']}")

    assert response.status_code == 200
    job = response.json()
    assert job["jobId"] == created["jobId"]
    assert job["userInstruction"] == "脸部再亮一点"
    assert job["status"] == "succeeded"


def test_create_retouch_job_rejects_missing_base_image(client: TestClient) -> None:
    _, plan = _analysis_and_plan(client)

    response = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": "img_missing", "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Base image not found."
