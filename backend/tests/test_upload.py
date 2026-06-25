from fastapi.testclient import TestClient

from app.core.config import Settings
from app.services.storage import StorageService
from tests.conftest import make_image_bytes


def test_upload_accepts_png(client: TestClient, temp_settings: Settings) -> None:
    image = make_image_bytes()

    response = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", image, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imageId"].startswith("img_")
    assert payload["filename"].endswith(".png")
    assert payload["contentType"] == "image/png"
    assert payload["width"] == 32
    assert payload["height"] == 24
    assert payload["sizeBytes"] == len(image)
    assert payload["url"].startswith("/data/uploads/")
    assert (temp_settings.uploads_dir / payload["filename"]).exists()


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/photos/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only JPG, PNG, and WebP images are supported."


async def _save_with_tiny_limit(settings: Settings) -> None:
    from starlette.datastructures import UploadFile

    service = StorageService(settings)
    upload = UploadFile(
        filename="large.png",
        file=__import__("io").BytesIO(make_image_bytes(size=(64, 64))),
        headers={"content-type": "image/png"},
    )
    await service.save_upload(upload)


def test_storage_enforces_size_limit(temp_settings: Settings) -> None:
    import asyncio
    from fastapi import HTTPException

    tiny_settings = Settings(data_dir=temp_settings.data_dir, max_upload_bytes=10)

    try:
        asyncio.run(_save_with_tiny_limit(tiny_settings))
    except HTTPException as exc:
        assert exc.status_code == 413
        assert exc.detail == "Image must be 10MB or smaller."
    else:
        raise AssertionError("Expected size limit failure")
