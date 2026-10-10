from __future__ import annotations

from pathlib import Path

from app.core.errors import bad_request
from app.domains.base import PhotoAnalyzer, RetouchPlanner
from app.domains.detector import AUTO_DOMAIN_TYPE, detect_domain_type
from app.domains.general.analyzer import GeneralAnalyzer
from app.domains.general.planner import GeneralPlanner
from app.domains.portrait.analyzer import PortraitAnalyzer
from app.domains.portrait.planner import PortraitPlanner
from app.domains.portrait.style_vocabulary import StyleVocabulary
from app.domains.portrait.stylist import PortraitStylist


class DomainRegistry:
    def __init__(self, vocabulary: StyleVocabulary | None = None) -> None:
        general_analyzer = GeneralAnalyzer()
        general_planner = GeneralPlanner()
        self.style_vocabulary = vocabulary or StyleVocabulary()
        self._analyzers: dict[str, PhotoAnalyzer] = {
            "portrait": PortraitAnalyzer(),
            "general": general_analyzer,
            "landscape": general_analyzer,
            "product": general_analyzer,
        }
        self._planners: dict[str, RetouchPlanner] = {
            "portrait": PortraitPlanner(),
            "general": general_planner,
            "landscape": general_planner,
            "product": general_planner,
        }
        # The stylist composes complete looks; today only portraits carry the
        # observable FaceFacts it needs, so it is registered for portraits.
        self._stylists: dict[str, PortraitStylist] = {
            "portrait": PortraitStylist(self.style_vocabulary),
        }

    def resolve_domain_type(self, domain_type: str, image_path: Path) -> str:
        if domain_type == AUTO_DOMAIN_TYPE:
            return detect_domain_type(image_path)
        return domain_type

    def get_analyzer(self, domain_type: str) -> PhotoAnalyzer:
        analyzer = self._analyzers.get(domain_type)
        if analyzer is None:
            raise bad_request(f"Unsupported domain type: {domain_type}")
        return analyzer

    def get_planner(self, domain_type: str) -> RetouchPlanner:
        planner = self._planners.get(domain_type)
        if planner is None:
            raise bad_request(f"Unsupported domain type: {domain_type}")
        return planner

    def get_stylist(self, domain_type: str) -> PortraitStylist:
        stylist = self._stylists.get(domain_type)
        if stylist is None:
            raise bad_request(
                f"Styling is only available for portraits, not: {domain_type}"
            )
        return stylist
