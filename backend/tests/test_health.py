from fastapi.testclient import TestClient

from app.main import create_app


def test_health_check() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Photo Retouch Agent API",
        "version": "0.2.0",
    }
