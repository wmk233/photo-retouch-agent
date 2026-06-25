from fastapi.testclient import TestClient

from tests.conftest import make_image_bytes


def _upload_sample(client: TestClient) -> str:
    response = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_image_bytes(size=(640, 960)), "image/png")},
    )
    assert response.status_code == 200
    return response.json()["imageId"]


def test_analyze_photo_returns_portrait_analysis(client: TestClient) -> None:
    image_id = _upload_sample(client)

    response = client.post("/api/photos/analyze", json={"imageId": image_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["imageId"] == image_id
    assert payload["domainType"] == "portrait"
    assert payload["sceneType"] == "人像 / 自拍"
    assert payload["subjects"]["count"] == 1
    assert payload["portraitSuggestions"]
    assert payload["recommendedStyles"]


def test_analyze_photo_returns_404_for_missing_image(client: TestClient) -> None:
    response = client.post("/api/photos/analyze", json={"imageId": "img_missing"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found."


def test_retouch_plans_returns_three_portrait_plans(client: TestClient) -> None:
    image_id = _upload_sample(client)
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id}).json()

    response = client.post("/api/retouch/plans", json={"analysis": analysis})

    assert response.status_code == 200
    plans = response.json()
    assert [plan["planId"] for plan in plans] == ["natural", "avatar", "mood"]
    assert all(plan["domainType"] == "portrait" for plan in plans)
    assert all(plan["editPrompt"] for plan in plans)
    assert plans[0]["title"] == "自然美化"
