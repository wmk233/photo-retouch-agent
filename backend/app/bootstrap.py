from __future__ import annotations

from dataclasses import dataclass, field

from app.application.ports import AnalysisCache, JobDispatcher
from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.domains.registry import DomainRegistry
from app.infrastructure.analysis_cache import InMemoryAnalysisCache
from app.infrastructure.job_store import JobStore
from app.infrastructure.storage import StorageService
from app.providers.factory import ImageProviderFactory
from app.services.task_queue import TaskQueue
from app.worker.in_process import InProcessJobDispatcher


@dataclass
class AppContainer:
    settings: Settings
    storage: StorageService
    jobs: JobStore
    image_providers: ImageProviderFactory
    brains: AgentBrainFactory
    domains: DomainRegistry
    task_queue: TaskQueue
    job_dispatcher: JobDispatcher
    analysis_cache: AnalysisCache = field(default_factory=InMemoryAnalysisCache)


def build_container(settings: Settings) -> AppContainer:
    task_queue = TaskQueue(max_concurrent=settings.max_concurrent_jobs)
    return AppContainer(
        settings=settings,
        storage=StorageService(settings),
        jobs=JobStore(settings),
        image_providers=ImageProviderFactory(settings),
        brains=AgentBrainFactory(settings),
        domains=DomainRegistry(),
        task_queue=task_queue,
        job_dispatcher=InProcessJobDispatcher(task_queue),
        analysis_cache=InMemoryAnalysisCache(
            ttl_seconds=settings.analysis_cache_ttl_seconds,
            max_entries=settings.analysis_cache_max_entries,
        ),
    )
