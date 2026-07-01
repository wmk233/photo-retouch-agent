import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    app_name: str = "Photo Retouch Agent API"
    app_version: str = "0.1.0"
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    data_dir: Path = PROJECT_ROOT / "data"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    image_provider: str = field(
        default_factory=lambda: os.getenv("PHOTO_AGENT_IMAGE_PROVIDER", "mock").strip().lower()
    )
    dashscope_api_key: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_API_KEY"),
        repr=False,
    )
    dashscope_workspace_id: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_WORKSPACE_ID")
    )
    dashscope_endpoint: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_ENDPOINT")
    )
    dashscope_region: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_REGION", "beijing").strip().lower()
    )
    dashscope_image_model: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_IMAGE_MODEL", "qwen-image-2.0-pro"
        ).strip()
    )
    provider_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("PHOTO_AGENT_PROVIDER_TIMEOUT", "180"))
    )
    provider_download_limit_bytes: int = 30 * 1024 * 1024

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
