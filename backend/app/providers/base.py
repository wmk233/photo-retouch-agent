from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.schemas.retouch import RetouchPlan


class ImageEditProvider(Protocol):
    provider_name: str
    model_name: str
    output_extension: str

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> None:
        """Edit source image and write the result to output_path."""
