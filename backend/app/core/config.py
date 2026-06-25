from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Photo Retouch Agent API"
    app_version: str = "0.1.0"
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    data_dir: Path = Path(__file__).resolve().parents[3] / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


settings = Settings()
