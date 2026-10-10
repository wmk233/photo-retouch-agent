from __future__ import annotations

import hashlib
from pathlib import Path

from app.application.ports import (
    AnalysisCatalog,
    AssetReader,
    PlanCatalog,
    PromptAnalyzer,
    StylingCatalog,
)
from app.brains.base import PromptBrain
from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan
from app.domain.styling import StylingProposal
from app.domains.registry import DomainRegistry
from app.schemas.mappers import (
    analysis_to_domain,
    analysis_to_schema,
    plan_to_domain,
)
from app.services.storage import StorageService


class StorageAssetReader(AssetReader):
    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    def resolve_path(self, image_id: str) -> Path | None:
        return self._storage.resolve_image_path(image_id)


class RegistryAnalysisCatalog(AnalysisCatalog):
    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        resolved = self._registry.resolve_domain_type(domain_type, image_path)
        analyzer = self._registry.get_analyzer(resolved)
        return analysis_to_domain(analyzer.analyze(image_id, image_path))


class BrainPromptAnalyzer(PromptAnalyzer):
    def __init__(self, brain: PromptBrain) -> None:
        self._brain = brain

    @property
    def cache_identity(self) -> str | None:
        """Identity of the model that produced the analysis.

        Used as the analysis-cache discriminator so switching providers (or
        keys) can never reuse another model's result. The key itself is only
        ever hashed, never stored or logged.
        """

        provider = getattr(self._brain, "provider_name", None)
        model = getattr(self._brain, "model_name", None)
        vision = getattr(self._brain, "vision_mode", None)
        if not provider or not model:
            return None
        api_key = getattr(self._brain, "_api_key", "") or ""
        fingerprint = (
            hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
            if api_key
            else "no-key"
        )
        return f"{provider}:{model}:{vision}:{fingerprint}"

    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        enriched = await self._brain.analyze(
            image_id,
            image_path,
            analysis_to_schema(baseline),
        )
        return analysis_to_domain(enriched)


class RegistryPlanCatalog(PlanCatalog):
    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def create_plans(self, analysis: PhotoAnalysis) -> tuple[RetouchPlan, ...]:
        planner = self._registry.get_planner(analysis.domain_type)
        plans = planner.create_plans(analysis_to_schema(analysis))
        return tuple(plan_to_domain(plan) for plan in plans)


class RegistryStylingCatalog(StylingCatalog):
    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def create_styling_proposals(
        self,
        analysis: PhotoAnalysis,
    ) -> tuple[StylingProposal, ...]:
        stylist = self._registry.get_stylist(analysis.domain_type)
        return stylist.create_proposals(analysis)
