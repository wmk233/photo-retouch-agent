from app.core.errors import bad_request
from app.domains.base import PhotoAnalyzer, RetouchPlanner
from app.domains.portrait.analyzer import PortraitAnalyzer
from app.domains.portrait.planner import PortraitPlanner


class DomainRegistry:
    def __init__(self) -> None:
        self._analyzers: dict[str, PhotoAnalyzer] = {
            "portrait": PortraitAnalyzer(),
        }
        self._planners: dict[str, RetouchPlanner] = {
            "portrait": PortraitPlanner(),
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
