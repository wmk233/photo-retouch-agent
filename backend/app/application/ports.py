from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Protocol, Sequence

from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan
from app.domain.styling import StylingProposal


class AssetReader(Protocol):
    def resolve_path(self, image_id: str) -> Path | None:
        """Resolve an asset to a local readable path for processing."""


class AnalysisCatalog(Protocol):
    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        """Run the deterministic/domain-specific baseline analyzer."""


class PromptAnalyzer(Protocol):
    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        """Enrich a baseline analysis with the selected AI brain."""


class AnalysisCache(Protocol):
    """Stores a finished analysis so a repeat call is not billed twice."""

    def get(self, key: str) -> PhotoAnalysis | None:
        """Return a still-valid cached analysis, or ``None``."""

    def put(self, key: str, analysis: PhotoAnalysis) -> None:
        """Store an analysis under ``key``."""


class PlanCatalog(Protocol):
    def create_plans(self, analysis: PhotoAnalysis) -> Sequence[RetouchPlan]:
        """Create retouch plan candidates for a domain analysis."""


class StylingCatalog(Protocol):
    def create_styling_proposals(
        self,
        analysis: PhotoAnalysis,
    ) -> Sequence[StylingProposal]:
        """Design complete looks for the person in a portrait analysis."""


JobExecutor = Callable[[str], Awaitable[object]]


class JobDispatcher(Protocol):
    async def dispatch(self, job_id: str, executor: JobExecutor) -> None:
        """Submit a job without coupling callers to the execution transport."""
