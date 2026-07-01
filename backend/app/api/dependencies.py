from typing import Annotated

from fastapi import Depends

from app.services.storage import StorageService, storage_service
from app.services.job_store import JobStore, job_store
from app.services.retouch_service import RetouchService
from app.providers.factory import ImageProviderFactory, provider_factory
from app.brains.factory import AgentBrainFactory, agent_brain_factory


def get_storage_service() -> StorageService:
    return storage_service


def get_job_store() -> JobStore:
    return job_store


def get_provider_factory() -> ImageProviderFactory:
    return provider_factory


def get_agent_brain_factory() -> AgentBrainFactory:
    return agent_brain_factory


def get_retouch_service(
    storage: Annotated[StorageService, Depends(get_storage_service)],
    jobs: Annotated[JobStore, Depends(get_job_store)],
) -> RetouchService:
    return RetouchService(storage, jobs)
