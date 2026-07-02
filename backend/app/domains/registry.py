from __future__ import annotations
from app.core.errors import bad_request
from app.domains.base import PhotoAnalyzer, RetouchPlanner
from app.domains.general.analyzer import GeneralAnalyzer
from app.domains.general.planner import GeneralPlanner
from app.domains.portrait.analyzer import PortraitAnalyzer
from app.domains.portrait.planner import PortraitPlanner


class DomainRegistry:
    def __init__(self) -> None:
        general_analyzer = GeneralAnalyzer()
        general_planner = GeneralPlanner()
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


domain_registry = DomainRegistry()
