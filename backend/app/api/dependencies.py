from typing import Annotated

from fastapi import Depends, Request

from app.api.application_adapters import (
    RegistryAnalysisCatalog,
    RegistryPlanCatalog,
    RegistryStylingCatalog,
    StorageAssetReader,
)
from app.application.use_cases import (
    AnalyzePhoto,
    CreateRetouchPlans,
    CreateStylingProposals,
)
from app.application.ports import AnalysisCache, JobDispatcher
from app.bootstrap import AppContainer
from app.domains.registry import DomainRegistry
from app.infrastructure.storage import StorageService
from app.infrastructure.job_store import JobStore
from app.services.retouch_service import RetouchService
from app.services.studio_interrogate import StudioInterrogateService
from app.services.studio_service import StudioService
from app.providers.factory import ImageProviderFactory
from app.brains.factory import AgentBrainFactory
from app.services.task_queue import TaskQueue


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_storage_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> StorageService:
    return container.storage


def get_job_store(
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobStore:
    return container.jobs


def get_provider_factory(
    container: Annotated[AppContainer, Depends(get_container)],
) -> ImageProviderFactory:
    return container.image_providers


def get_agent_brain_factory(
    container: Annotated[AppContainer, Depends(get_container)],
) -> AgentBrainFactory:
    return container.brains


def get_domain_registry(
    container: Annotated[AppContainer, Depends(get_container)],
) -> DomainRegistry:
    return container.domains


def get_task_queue(
    container: Annotated[AppContainer, Depends(get_container)],
) -> TaskQueue:
    return container.task_queue


def get_job_dispatcher(
    container: Annotated[AppContainer, Depends(get_container)],
) -> JobDispatcher:
    return container.job_dispatcher


def get_retouch_service(
    container: Annotated[AppContainer, Depends(get_container)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    jobs: Annotated[JobStore, Depends(get_job_store)],
) -> RetouchService:
    return RetouchService(storage, jobs, settings=container.settings)


def get_analysis_cache(
    container: Annotated[AppContainer, Depends(get_container)],
) -> AnalysisCache:
    return container.analysis_cache


def get_analyze_photo_use_case(
    storage: Annotated[StorageService, Depends(get_storage_service)],
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
    cache: Annotated[AnalysisCache, Depends(get_analysis_cache)],
) -> AnalyzePhoto:
    return AnalyzePhoto(
        assets=StorageAssetReader(storage),
        analyzers=RegistryAnalysisCatalog(domains),
        cache=cache,
    )


def get_create_plans_use_case(
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> CreateRetouchPlans:
    return CreateRetouchPlans(RegistryPlanCatalog(domains))


def get_create_styling_proposals_use_case(
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> CreateStylingProposals:
    return CreateStylingProposals(RegistryStylingCatalog(domains))


def get_studio_service(
    container: Annotated[AppContainer, Depends(get_container)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
) -> StudioService:
    return StudioService(container.settings, storage=storage)


def get_studio_interrogate_service(
    container: Annotated[AppContainer, Depends(get_container)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
) -> StudioInterrogateService:
    return StudioInterrogateService(container.settings, storage=storage)
