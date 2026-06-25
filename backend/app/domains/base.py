from pathlib import Path
from typing import Protocol

from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class PhotoAnalyzer(Protocol):
    def analyze(self, image_id: str, image_path: Path) -> PhotoAnalysis:
        """Analyze an image and return structured domain-specific observations."""


class RetouchPlanner(Protocol):
    def create_plans(self, analysis: PhotoAnalysis) -> list[RetouchPlan]:
        """Create candidate retouch plans from an analysis result."""
