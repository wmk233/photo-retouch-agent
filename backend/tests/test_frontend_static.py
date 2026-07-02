from fastapi.testclient import TestClient


def test_frontend_index_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Photo Retouch Agent" in response.text
    assert './js/main.js' in response.text
    assert 'id="brainProviderSelect"' in response.text
    assert 'id="actionProviderSelect"' in response.text
    assert 'id="workflowGate"' in response.text
    assert '<option value="">请选择</option>' in response.text


def test_frontend_js_is_served(client: TestClient) -> None:
    response = client.get("/js/main.js")

    assert response.status_code == 200
    assert "createJob" in response.text
