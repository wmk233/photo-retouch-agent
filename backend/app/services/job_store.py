from pathlib import Path

from app.core.config import Settings, settings
from app.core.errors import not_found
from app.schemas.job import RetouchJob
from app.services.storage import ensure_data_dirs


class JobStore:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        ensure_data_dirs(config)

    def save(self, job: RetouchJob) -> RetouchJob:
        path = self._job_path(job.job_id)
        path.write_text(job.model_dump_json(), encoding="utf-8")
        return job

    def get(self, job_id: str) -> RetouchJob:
        path = self._job_path(job_id)
        if not path.exists():
            raise not_found("Retouch job not found.")
        return RetouchJob.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[RetouchJob]:
        paths = sorted(
            self.config.jobs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        jobs: list[RetouchJob] = []
        for path in paths[offset : offset + limit]:
            try:
                jobs.append(RetouchJob.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return jobs

    def _job_path(self, job_id: str) -> Path:
        return self.config.jobs_dir / f"{job_id}.json"


job_store = JobStore()
