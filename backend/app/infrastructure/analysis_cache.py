from __future__ import annotations

import time
from collections import OrderedDict

from app.domain.analysis import PhotoAnalysis


class InMemoryAnalysisCache:
    """A bounded, expiring cache of finished photo analyses.

    ``POST /api/photos/analyze`` is a billable *vision* call. When the same
    image is analysed twice with the same provider, model and key, the second
    call can return the stored analysis instead of paying the provider again.

    Entries older than ``ttl_seconds`` are treated as missing and evicted;
    once more than ``max_entries`` are stored the least recently used entry is
    dropped. A non-positive ``ttl_seconds`` or ``max_entries`` disables the
    cache entirely, so an operator can always opt out of reuse.
    """

    def __init__(self, ttl_seconds: float = 900.0, max_entries: int = 64) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[str, tuple[float, PhotoAnalysis]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds > 0 and self._max_entries > 0

    def get(self, key: str) -> PhotoAnalysis | None:
        if not self.enabled:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, analysis = entry
        if self._is_expired(stored_at):
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return analysis

    def put(self, key: str, analysis: PhotoAnalysis) -> None:
        if not self.enabled:
            return
        self._entries[key] = (time.monotonic(), analysis)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def _is_expired(self, stored_at: float) -> bool:
        return (time.monotonic() - stored_at) > self._ttl_seconds
