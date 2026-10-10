from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


RATE_LIMITED_PREFIX = "/api"

class TokenBucket:
    def __init__(self, rate: float = 10, burst: int = 20) -> None:
        self._rate = rate
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (float(burst), time.monotonic())
        )

    def consume(self, key: str) -> bool:
        tokens, last = self._buckets[key]
        now = time.monotonic()
        tokens = min(self._burst, tokens + (now - last) * self._rate)
        if tokens < 1:
            return False
        self._buckets[key] = (tokens - 1, now)
        return True


_buckets: dict[int, TokenBucket] = {}


def _get_bucket(settings) -> TokenBucket:
    bucket_id = id(settings)
    if bucket_id not in _buckets:
        rate = float(settings.rate_limit_per_second)
        _buckets[bucket_id] = TokenBucket(rate=rate, burst=int(rate * 2))
    return _buckets[bucket_id]


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    return ip or request.client.host if request.client else "unknown"


async def rate_limit_middleware(request: Request, call_next):
    app_settings = request.app.state.settings
    rate = app_settings.rate_limit_per_second
    # Static assets (the HTML/CSS/ES modules of the local UI) must never burn
    # buckets: a single page load is ~25 requests and would starve the API.
    if rate <= 0 or not request.url.path.startswith(RATE_LIMITED_PREFIX):
        return await call_next(request)
    if not _get_bucket(app_settings).consume(_client_key(request)):
        # Raising HTTPException from user middleware escapes FastAPI's exception
        # handlers and surfaces as a misleading 500. Return the real status.
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests"},
            headers={"Retry-After": "1"},
        )
    return await call_next(request)
