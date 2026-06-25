from app.services.storage import StorageService, storage_service


def get_storage_service() -> StorageService:
    return storage_service
