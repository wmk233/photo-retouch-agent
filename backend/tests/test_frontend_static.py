from fastapi.testclient import TestClient


def test_frontend_index_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Photo Retouch Agent" in response.text
    assert './js/main.js' in response.text
    assert 'data-mode="local"' in response.text
    assert 'data-mode="ai"' in response.text
    assert 'data-category="skin"' in response.text
    assert 'id="compareRange"' in response.text
    assert 'id="generateButton"' in response.text


def test_frontend_js_is_served(client: TestClient) -> None:
    response = client.get("/js/main.js")

    assert response.status_code == 200
    assert "const categories" in response.text
    assert "updatePreview" in response.text
    assert "applyPreset" in response.text

    engine_response = client.get("/js/local-retouch.mjs")
    assert engine_response.status_code == 200
    assert "renderLocalRetouch" in engine_response.text
    assert "createRenderRecipe" in engine_response.text

    ai_response = client.get("/js/ai-retouch.mjs")
    assert ai_response.status_code == 200
    assert "executeAiRetouch" in ai_response.text
    assert "buildRetouchInstruction" in ai_response.text

    export_response = client.get("/js/export-image.mjs")
    assert export_response.status_code == 200
    assert "canvasToBlob" in export_response.text
    assert "triggerBlobDownload" in export_response.text
