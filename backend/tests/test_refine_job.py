from fastapi.testclient import TestClient

from tests.conftest import make_image_bytes


def _create_completed_job(client: TestClient) -> dict:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_image_bytes(size=(640, 960)), "image/png")},
    )
    assert upload.status_code == 200
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    job = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plans.json()[0], "userInstruction": ""},
    )
    assert job.status_code == 200
    return job.json()


def test_refine_job_uses_previous_output_as_base(client: TestClient) -> None:
    parent = _create_completed_job(client)
    previous_output_id = parent["outputImageIds"][0]

    response = client.post(
        f"/api/retouch/jobs/{parent['jobId']}/refine",
        json={"userInstruction": "再自然一点，肤色再亮一些"},
    )

    assert response.status_code == 200
    refined = response.json()
    assert refined["jobId"] != parent["jobId"]
    assert refined["sourceImageId"] == parent["sourceImageId"]
    assert refined["baseImageId"] == previous_output_id
    assert refined["status"] == "succeeded"
    assert refined["userInstruction"] == "再自然一点，肤色再亮一些"
    assert refined["outputImageIds"][0] != previous_output_id


def test_refine_missing_job_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/retouch/jobs/job_missing/refine",
        json={"userInstruction": "再亮一点"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Retouch job not found."
