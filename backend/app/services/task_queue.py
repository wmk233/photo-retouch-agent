from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger("photo_retouch.task_queue")


class TaskQueue:
    """In-process background task runner with concurrency control."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def enqueue(
        self,
        job_id: str,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        async def _run() -> None:
            async with self._semaphore:
                logger.info("task_started", job_id=job_id)
                try:
                    await coro_factory()
                except Exception:
                    logger.exception("task_failed", job_id=job_id)

        asyncio.create_task(_run())
        logger.info("task_enqueued", job_id=job_id)


_task_queue: TaskQueue | None = None


def get_task_queue(max_concurrent: int = 3) -> TaskQueue:
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(max_concurrent=max_concurrent)
    return _task_queue
