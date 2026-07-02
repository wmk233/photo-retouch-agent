from __future__ import annotations

from fastapi import Request, HTTPException

AUTH_REQUIRED_PREFIXES = (
    "/api/photos/upload",
    "/api/retouch/jobs",
)


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    requires_auth = any(path.startswith(prefix) for prefix in AUTH_REQUIRED_PREFIXES)
    app_settings = request.app.state.settings

    if requires_auth and app_settings.api_key:
        api_key = request.headers.get("X-API-Key", "")
        if api_key != app_settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return await call_next(request)
