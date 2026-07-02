from __future__ import annotations
from app.core.errors import bad_request


def require_model_selection(
    primary_name: str | None,
    legacy_name: str | None,
    layer_label: str,
) -> str:
    name = (primary_name or legacy_name or "").strip().lower()
    if not name or name == "auto":
        raise bad_request(f"Select an explicit {layer_label} model before running workflow.")
    return name
