"""Regression tests for portrait-vs-general domain routing.

The web UI used to send ``domainType: "general"`` for every upload, so a real
portrait never reached the portrait analyzer/planner. The fix lets the caller
ask the server to infer the domain (``"auto"``) without hardcoding ``portrait``:
a portrait must be routed into the portrait domain, and everything else must
stay out of it.
"""

from fastapi.testclient import TestClient

from tests.conftest import make_landscape_bytes, make_portrait_bytes


def _upload(client: TestClient, data: bytes, name: str) -> str:
    response = client.post(
        "/api/photos/upload",
        files={"file": (name, data, "image/png")},
    )
    assert response.status_code == 200
    return response.json()["imageId"]


def _analyze(client: TestClient, image_id: str, domain_type: str | None = None):
    payload: dict[str, str] = {"imageId": image_id}
    if domain_type is not None:
        payload["domainType"] = domain_type
    return client.post("/api/photos/analyze", json=payload)


def _plan_ids(client: TestClient, analysis: dict) -> list[str]:
    response = client.post("/api/retouch/plans", json={"analysis": analysis})
    assert response.status_code == 200
    return [plan["planId"] for plan in response.json()]


def test_auto_routes_a_portrait_photo_into_the_portrait_domain(
    client: TestClient,
) -> None:
    image_id = _upload(client, make_portrait_bytes(), "portrait.png")

    response = _analyze(client, image_id, "auto")

    assert response.status_code == 200
    payload = response.json()
    assert payload["domainType"] == "portrait"
    assert payload["sceneType"] == "人像 / 自拍"
    assert _plan_ids(client, payload) == ["natural", "avatar", "mood"]


def test_auto_keeps_a_non_portrait_photo_out_of_the_portrait_domain(
    client: TestClient,
) -> None:
    image_id = _upload(client, make_landscape_bytes(), "landscape.png")

    response = _analyze(client, image_id, "auto")

    assert response.status_code == 200
    payload = response.json()
    assert payload["domainType"] == "general"
    assert _plan_ids(client, payload) == ["natural", "clean", "mood"]


def test_auto_does_not_treat_tall_scenery_as_a_portrait(
    client: TestClient,
) -> None:
    image_id = _upload(
        client,
        make_landscape_bytes(size=(640, 960)),
        "tall-scenery.png",
    )

    response = _analyze(client, image_id, "auto")

    assert response.status_code == 200
    assert response.json()["domainType"] == "general"


def test_explicit_domain_type_still_overrides_detection(
    client: TestClient,
) -> None:
    portrait_id = _upload(client, make_portrait_bytes(), "portrait.png")
    forced_general = _analyze(client, portrait_id, "general")
    assert forced_general.status_code == 200
    assert forced_general.json()["domainType"] == "general"

    landscape_id = _upload(client, make_landscape_bytes(), "landscape.png")
    forced_portrait = _analyze(client, landscape_id, "portrait")
    assert forced_portrait.status_code == 200
    assert forced_portrait.json()["domainType"] == "portrait"


def test_unknown_domain_type_is_rejected(client: TestClient) -> None:
    image_id = _upload(client, make_portrait_bytes(), "portrait.png")

    response = _analyze(client, image_id, "banana")

    assert response.status_code == 400
    assert "Unsupported domain type" in response.json()["detail"]
