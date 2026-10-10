from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from app.application.commands import AnalyzePhotoCommand
from app.application.ports import (
    AnalysisCache,
    AnalysisCatalog,
    AssetReader,
    PlanCatalog,
    PromptAnalyzer,
    StylingCatalog,
)
from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan
from app.domain.styling import StylingProposal


class AssetNotFoundError(LookupError):
    pass


class AnalyzePhoto:
    def __init__(
        self,
        assets: AssetReader,
        analyzers: AnalysisCatalog,
        cache: AnalysisCache | None = None,
    ) -> None:
        self._assets = assets
        self._analyzers = analyzers
        self._cache = cache

    async def execute(
        self,
        command: AnalyzePhotoCommand,
        brain: PromptAnalyzer,
    ) -> PhotoAnalysis:
        image_path = self._assets.resolve_path(command.image_id)
        if image_path is None:
            raise AssetNotFoundError(command.image_id)

        cache_key = self._cache_key(command, image_path, brain)
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Serve the exact analysis the provider already produced. Only
                # the identity flag and the current image id change, so a
                # repeat upload of the same bytes never bills the provider
                # again. Nothing about the person is invented: these are the
                # provider's own words, reused verbatim.
                return replace(
                    cached,
                    image_id=command.image_id,
                    analysis_reused=True,
                )

        baseline = self._analyzers.analyze(
            command.image_id,
            image_path,
            command.domain_type,
        )
        analysis = await brain.enrich(command.image_id, image_path, baseline)
        if cache_key is not None:
            self._cache.put(cache_key, analysis)
        return analysis

    def _cache_key(
        self,
        command: AnalyzePhotoCommand,
        image_path: Path,
        brain: PromptAnalyzer,
    ) -> str | None:
        """A provider- and content-scoped key, or ``None`` if unsafe.

        The key is scoped to the *pixels* rather than the upload id: the
        frontend re-uploads the file on every design attempt, so an id-based
        key would never hit. The enrichment also depends on the chosen model,
        so the brain's identity (provider, model, vision mode, key
        fingerprint) is part of the key. A brain that cannot describe itself
        is never cached.
        """

        if self._cache is None:
            return None
        identity = getattr(brain, "cache_identity", None)
        if not identity:
            return None
        fingerprint = _content_fingerprint(image_path)
        if fingerprint is None:
            return None
        return f"{fingerprint}|{command.domain_type}|{identity}"


class CreateRetouchPlans:
    def __init__(self, planners: PlanCatalog) -> None:
        self._planners = planners

    def execute(self, analysis: PhotoAnalysis) -> tuple[RetouchPlan, ...]:
        return tuple(self._planners.create_plans(analysis))


class CreateStylingProposals:
    """Design three complete looks for the person in an analysed portrait.

    This is pure computation over the analysis (no network, no image model),
    so it never incurs a paid request.
    """

    def __init__(self, stylists: StylingCatalog) -> None:
        self._stylists = stylists

    def execute(self, analysis: PhotoAnalysis) -> tuple[StylingProposal, ...]:
        return tuple(self._stylists.create_styling_proposals(analysis))


def _content_fingerprint(image_path: Path) -> str | None:
    """A short, stable fingerprint of the image bytes, or ``None``.

    Identical bytes must reuse the same analysis; different bytes must never
    share one. The digest is truncated - it only needs to separate images, not
    resist attack.
    """

    try:
        digest = hashlib.sha256()
        with open(image_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()[:32]
