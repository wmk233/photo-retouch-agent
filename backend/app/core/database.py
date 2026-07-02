from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.job import RetouchJob

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retouch_jobs (
    job_id           TEXT PRIMARY KEY,
    source_image_id  TEXT NOT NULL,
    base_image_id    TEXT,
    plan_id          TEXT NOT NULL,
    plan_json        TEXT NOT NULL,
    user_instruction TEXT NOT NULL DEFAULT '',
    model_provider   TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    brain_provider   TEXT NOT NULL,
    brain_model      TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    output_image_ids TEXT,
    output_urls      TEXT,
    error_message    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON retouch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON retouch_jobs(created_at DESC);
"""


class Database:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path or settings.database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_job(self, job: RetouchJob) -> RetouchJob:
        self._conn.execute(
            """INSERT INTO retouch_jobs (
                job_id, source_image_id, base_image_id, plan_id, plan_json,
                user_instruction, model_provider, model_name,
                brain_provider, brain_model, status,
                output_image_ids, output_urls, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.job_id,
                job.source_image_id,
                job.base_image_id,
                job.plan_id,
                job.plan.model_dump_json(by_alias=True) if job.plan else "",
                job.user_instruction,
                job.model_provider or "",
                job.model_name or "",
                job.brain_provider or "",
                job.brain_model or "",
                job.status,
                json.dumps(job.output_image_ids),
                json.dumps(job.output_urls),
                job.error_message,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
            ),
        )
        self._conn.commit()
        return job

    def get_job(self, job_id: str) -> RetouchJob | None:
        row = self._conn.execute(
            "SELECT * FROM retouch_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        plan: Any = None,
        output_image_ids: list[str] | None = None,
        output_urls: list[str] | None = None,
        error_message: str | None = None,
    ) -> RetouchJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        if status is not None:
            job.status = status
        if plan is not None:
            job.plan = plan
        if output_image_ids is not None:
            job.output_image_ids = output_image_ids
        if output_urls is not None:
            job.output_urls = output_urls
        if error_message is not None:
            job.error_message = error_message

        from datetime import datetime, timezone

        job.updated_at = datetime.now(timezone.utc)

        self._conn.execute(
            """UPDATE retouch_jobs SET
                status=?, plan_json=?, output_image_ids=?, output_urls=?,
                error_message=?, updated_at=?
            WHERE job_id=?""",
            (
                job.status,
                job.plan.model_dump_json(by_alias=True) if job.plan else "",
                json.dumps(job.output_image_ids),
                json.dumps(job.output_urls),
                job.error_message,
                job.updated_at.isoformat(),
                job_id,
            ),
        )
        self._conn.commit()
        return job

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[RetouchJob]:
        rows = self._conn.execute(
            "SELECT * FROM retouch_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> RetouchJob:
        from datetime import datetime

        from app.schemas.retouch import RetouchPlan

        plan = None
        if row["plan_json"]:
            try:
                plan = RetouchPlan.model_validate_json(row["plan_json"])
            except Exception:
                plan = None

        def _parse_json(val: str | None) -> list[str]:
            if not val:
                return []
            try:
                return json.loads(val)
            except Exception:
                return []

        return RetouchJob(
            job_id=row["job_id"],
            source_image_id=row["source_image_id"],
            base_image_id=row["base_image_id"],
            plan_id=row["plan_id"],
            plan=plan,
            user_instruction=row["user_instruction"],
            model_provider=row["model_provider"],
            model_name=row["model_name"],
            brain_provider=row["brain_provider"],
            brain_model=row["brain_model"],
            status=row["status"],
            output_image_ids=_parse_json(row["output_image_ids"]),
            output_urls=_parse_json(row["output_urls"]),
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
