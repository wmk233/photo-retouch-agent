# Photo Retouch Agent v0.3.0 — Architecture Design

## 1. Overview

Photo Retouch Agent 是一个照片美化平台，支持**本地实时预览**和**AI 云端精修**两种工作模式。v0.2.0 (mvp-demo-code) 已完成核心闭环，v0.3.0 的目标是解决生产化瓶颈：异步化、持久化、可观测性、以及前端工程化。

### 1.1 Goals

- 全链路异步化，支持并发请求
- 任务队列解耦长耗时 AI 调用
- 结构化持久化替代 JSON 文件存储
- 前端模块化拆分，消除硬编码数据
- 基础认证与频率限制
- 可观测性（日志、指标）

### 1.2 Non-Goals

- 不引入微服务拆分（保持单体部署）
- 不支持多人协作/多租户
- 不做移动端适配
- 不引入真实人脸检测模型（留到 v0.4.0）

---

## 2. Current State Analysis (v0.2.0)

### 2.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│ Frontend (vanilla ES modules, served by FastAPI)         │
│                                                          │
│  index.html                                              │
│  ├── js/main.js          (700 lines — state + UI + events)│
│  ├── js/api.js           (HTTP client)                   │
│  ├── js/ai-retouch.mjs   (AI workflow orchestrator)      │
│  ├── js/local-retouch.mjs(Canvas 2D renderer)            │
│  └── js/export-image.mjs (download helper)               │
│                                                          │
│  css/styles.css          (1669 lines)                    │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP (sync)
┌──────────────────────▼───────────────────────────────────┐
│ Backend (FastAPI, app/main.py)                           │
│                                                          │
│  api/routes/                                             │
│  ├── health.py         GET  /api/health                  │
│  ├── photos.py         POST /api/photos/upload           │
│  ├── analyze.py        POST /api/photos/analyze          │
│  └── retouch.py        POST /api/retouch/plans           │
│                        POST /api/retouch/jobs             │
│                        GET  /api/retouch/jobs/{id}        │
│                        POST /api/retouch/jobs/{id}/refine │
│                        GET  /api/retouch/providers        │
│                                                          │
│  services/                                               │
│  ├── retouch_service.py  (orchestrator, sync)            │
│  ├── storage.py          (file I/O)                      │
│  └── job_store.py        (JSON file persistence)         │
│                                                          │
│  domains/              brains/          providers/       │
│  ├── base.py           ├── base.py      ├── base.py      │
│  ├── registry.py       ├── factory.py   ├── factory.py   │
│  ├── portrait/         ├── local.py     ├── mock.py      │
│  │   ├── analyzer.py   └── openai_      ├── qwen.py      │
│  │   └── planner.py        compatible   └── seedream.py  │
│  └── general/              .py                           │
│      ├── analyzer.py                                     │
│      └── planner.py                                      │
│                                                          │
│  schemas/               core/                            │
│  ├── photo.py           ├── config.py (Settings)         │
│  ├── analysis.py        └── errors.py                    │
│  ├── retouch.py                                          │
│  └── job.py                                              │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Key Problems Identified

| ID | Problem | Severity | Impact |
|----|---------|----------|--------|
| P1 | 全链路同步 I/O (`httpx.Client`) | Critical | FastAPI event loop 阻塞，无法并发 |
| P2 | JSON 文件持久化 (`data/jobs/*.json`) | High | 无查询/索引/并发控制 |
| P3 | `main.js` 700 行单体文件 | Medium | 维护困难，状态管理混乱 |
| P4 | 硬编码示例图片路径 | Medium | 新环境无法开箱即用 |
| P5 | 无认证/频率限制 | High | 生产环境 API 滥用风险 |
| P6 | 无日志/可观测性 | High | 故障排查困难 |
| P7 | 长任务阻塞 HTTP 响应 | High | AI 生成 30s+ 期间连接挂起 |
| P8 | `except Exception` 吞没错误 | Medium | 调试困难，根因丢失 |
| P9 | 遗留原型文件在根目录 | Low | 混淆 |

---

## 3. Target Architecture (v0.3.0)

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend (ES modules, modularized)                          │
│                                                             │
│  js/                                                        │
│  ├── app.js              (bootstrap, ~80 lines)             │
│  ├── state.js            (reactive state store, ~120 lines) │
│  ├── ui/                                                     │
│  │   ├── category-panel.js                                 │
│  │   ├── tool-rail.js                                      │
│  │   ├── canvas-panel.js                                   │
│  │   ├── inspector-panel.js                                │
│  │   ├── ai-panel.js                                       │
│  │   └── header.js                                         │
│  ├── engine/                                                │
│  │   ├── local-retouch.mjs     (unchanged)                 │
│  │   └── export-image.mjs      (unchanged)                 │
│  ├── workflow/                                              │
│  │   └── ai-retouch.mjs        (unchanged)                 │
│  └── api.js                   (unchanged)                  │
│                                                             │
│  css/styles.css              (extracted)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/SSE
┌──────────────────────▼──────────────────────────────────────┐
│ Backend (FastAPI, fully async)                               │
│                                                              │
│  api/routes/                   (thin — delegate to services) │
│  api/dependencies.py           (DI, async)                   │
│  api/middleware/                                              │
│  ├── auth.py                   (API key validation)          │
│  ├── rate_limit.py             (token bucket)                │
│  └── logging.py                (request/response logging)    │
│                                                              │
│  services/                                                   │
│  ├── retouch_service.py        (async orchestrator)          │
│  ├── storage.py                (async file I/O)              │
│  ├── job_store.py              (SQLite via aiosqlite)        │
│  └── task_queue.py             (background task runner)      │
│                                                              │
│  domains/          brains/           providers/              │
│  (unchanged)       (async .complete) (async .edit_image)     │
│                                                              │
│  core/                                                       │
│  ├── config.py                                                │
│  ├── errors.py                                                │
│  ├── logging.py               (structured logging)           │
│  └── database.py              (SQLite connection pool)       │
│                                                              │
│  schemas/  (unchanged + new migration models)                │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Request Lifecycle (Async + Task Queue)

```
Client                     FastAPI                      Background
  │                          │                            │
  │  POST /api/retouch/jobs  │                            │
  │─────────────────────────>│                            │
  │                          │ validate + auth            │
  │                          │ create job (status=queued) │
  │                          │ enqueue task               │
  │                          │──┐                         │
  │  202 Accepted            │  │                         │
  │  {jobId, status:queued}  │  │                         │
  │<─────────────────────────│  │                         │
  │                          │  │  background task        │
  │                          │  │────────────────────────>│
  │                          │  │                         │ brain.optimize()
  │                          │  │                         │ provider.edit_image()
  │                          │  │                         │ update job status
  │                          │  │<────────────────────────│
  │                          │  │                         │
  │  GET /api/retouch/jobs/{id} (poll)                    │
  │─────────────────────────>│                            │
  │  200 {status:succeeded,  │                            │
  │       outputUrls:[...]}  │                            │
  │<─────────────────────────│                            │

  Alternative: SSE push when job completes (v0.3.1)
```

### 3.3 Technology Choices

| Layer | v0.2.0 | v0.3.0 | Reason |
|-------|--------|--------|--------|
| HTTP client | `httpx.Client` (sync) | `httpx.AsyncClient` | 非阻塞并发 |
| Persistence | JSON files | SQLite (`aiosqlite`) | 查询、索引、并发安全 |
| Task execution | Inline blocking | `asyncio.create_task` + polling | 不阻塞 HTTP 连接 |
| Logging | None | `structlog` | 结构化日志，可搜索 |
| Frontend state | Global mutable object | `Proxy`-based reactive store | 响应式更新，可测试 |
| Auth | None | Static API key (header) | 最小可用认证 |
| Rate limit | None | Token bucket (in-memory) | 防止 API 滥用 |

---

## 4. Detailed Module Design

### 4.1 Asynchronous Core

All I/O-bound operations become async:

```python
# v0.3.0 — brains/openai_compatible.py
class OpenAICompatiblePromptBrain:
    async def analyze(self, image_id, image_path, baseline) -> PhotoAnalysis:
        ...
        content = await self._complete(system_prompt, user_text, image_path)
        ...

    async def optimize(self, source_path, plan, user_instruction) -> RetouchPlan:
        ...
        content = await self._complete(system_prompt, user_text, source_path)
        ...

    async def _complete(self, system_prompt, user_text, image_path) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.endpoint, ...)
        return self._extract_content(response)
```

```python
# v0.3.0 — providers/qwen_image_provider.py
class QwenImageProvider:
    async def edit_image(self, source_path, output_path, plan, user_instruction) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.endpoint, ...)
            result = self._parse_response(response)
            image_url = self._extract_image_url(result)
            image_response = await client.get(image_url)
            await self._save_download(image_response, output_path)
```

**Protocol updates** — `brains/base.py` and `providers/base.py` change method signatures to `async`:

```python
class PromptBrain(Protocol):
    async def analyze(self, image_id: str, image_path: Path, baseline: PhotoAnalysis) -> PhotoAnalysis: ...
    async def optimize(self, source_path: Path, plan: RetouchPlan, user_instruction: str) -> RetouchPlan: ...
```

### 4.2 Task Queue

Instead of blocking the HTTP request for 30+ seconds, jobs run as background tasks:

```python
# services/task_queue.py
import asyncio
from app.core.database import get_db

class TaskQueue:
    """In-process background task runner for AI jobs."""

    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def enqueue(self, job_id: str, coro) -> None:
        async def _run():
            async with self._semaphore:
                db = await get_db()
                try:
                    await db.update_job_status(job_id, "running")
                    result = await coro
                    await db.update_job_status(job_id, "succeeded", result=result)
                except Exception as exc:
                    await db.update_job_status(job_id, "failed", error=str(exc))

        asyncio.create_task(_run())
```

Flow:
1. `POST /api/retouch/jobs` → creates job (status=`queued`), calls `task_queue.enqueue(...)`, returns 202
2. Client polls `GET /api/retouch/jobs/{id}` → returns current status
3. When complete → status=`succeeded` with `outputUrls`

**Semaphore limits concurrent AI calls** to prevent API rate limit hits and memory exhaustion.

### 4.3 Database Layer

Replace `data/jobs/job_*.json` with SQLite:

```sql
-- core/database.py — schema
CREATE TABLE IF NOT EXISTS retouch_jobs (
    job_id         TEXT PRIMARY KEY,
    source_image_id TEXT NOT NULL,
    base_image_id  TEXT,
    plan_id        TEXT NOT NULL,
    plan_json      TEXT NOT NULL,         -- JSON serialized RetouchPlan
    user_instruction TEXT NOT NULL DEFAULT '',
    model_provider  TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    brain_provider  TEXT NOT NULL,
    brain_model     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    output_image_ids TEXT,               -- JSON array
    output_urls     TEXT,                -- JSON array
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON retouch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON retouch_jobs(created_at DESC);
```

```python
# core/database.py
import aiosqlite

async def get_db() -> "Database":
    """Return a per-request database connection."""
    ...

class Database:
    async def create_job(self, job: RetouchJob) -> RetouchJob: ...
    async def get_job(self, job_id: str) -> RetouchJob | None: ...
    async def update_job_status(self, job_id: str, status: str, **kwargs) -> RetouchJob: ...
    async def list_jobs(self, limit: int = 20, offset: int = 0) -> list[RetouchJob]: ...
```

### 4.4 Authentication & Rate Limiting

**Authentication** — simple static API key for MVP:

```python
# api/middleware/auth.py
from fastapi import Request, HTTPException

AUTH_REQUIRED_PATHS = {"/api/photos/upload", "/api/retouch/jobs"}

async def auth_middleware(request: Request, call_next):
    if request.url.path in AUTH_REQUIRED_PATHS:
        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:  # from env PHOTO_AGENT_API_KEY
            raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(request)
```

Non-mutating endpoints (health, providers, job status) remain public for dev convenience.

**Rate limiting** — in-memory token bucket:

```python
# api/middleware/rate_limit.py
from collections import defaultdict
import time

class TokenBucket:
    def __init__(self, rate: int = 10, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (burst, time.monotonic()))

    def consume(self, key: str) -> bool:
        tokens, last = self._buckets[key]
        now = time.monotonic()
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1:
            return False
        self._buckets[key] = (tokens - 1, now)
        return True
```

### 4.5 Structured Logging

```python
# core/logging.py
import structlog

logger = structlog.get_logger()

# Usage in retouch_service.py
logger.info("retouch_job_started", job_id=job.job_id, provider=self.provider.provider_name)
logger.error("retouch_job_failed", job_id=job.job_id, error=str(exc), exc_info=True)
```

Log output in development: human-readable console. In production: JSON to stdout.

### 4.6 Error Handling

Replace bare `except Exception` with structured error capture:

```python
# services/retouch_service.py
class RetouchJobError(Exception):
    def __init__(self, message: str, job_id: str, phase: str, original: Exception | None = None):
        super().__init__(message)
        self.job_id = job_id
        self.phase = phase
        self.original = original

# In create_job:
try:
    effective_plan = await self.brain.optimize(...)
except PromptBrainError as exc:
    raise RetouchJobError(
        f"Brain optimization failed: {exc}",
        job_id=job.job_id,
        phase="optimize",
        original=exc,
    )
```

---

## 5. Frontend Refactor

### 5.1 Module Split

```
frontend/js/
├── app.js                 Bootstrap, wires modules together (~80 lines)
├── state.js               Reactive state via Proxy (~120 lines)
├── api.js                 Unchanged
├── ui/
│   ├── category-panel.js  Category tabs, tool search, tags, sliders
│   ├── tool-rail.js       Left sidebar tool categories
│   ├── canvas-panel.js    Canvas, compare slider, before/after overlay
│   ├── inspector-panel.js Face detection card, changes list, engine card
│   ├── ai-panel.js        AI mode: presets, prompt, model config, generate
│   └── header.js          Brand, file name, undo/reset/compare/export buttons
├── engine/
│   ├── local-retouch.mjs  Unchanged
│   └── export-image.mjs   Unchanged
└── workflow/
    └── ai-retouch.mjs     Unchanged
```

### 5.2 Reactive State Store

```javascript
// js/state.js
function createStore(initial) {
  const listeners = new Map();

  const state = new Proxy(initial, {
    set(target, key, value) {
      target[key] = value;
      (listeners.get(key) || []).forEach(fn => fn(value));
      (listeners.get('*') || []).forEach(fn => fn(key, value));
      return true;
    }
  });

  return {
    state,
    on(key, fn) {
      if (!listeners.has(key)) listeners.set(key, []);
      listeners.get(key).push(fn);
    },
  };
}

export const { state, on } = createStore({
  category: 'skin',
  mode: 'local',
  values: { /* 49 tool defaults */ },
  history: [],
  selectedPreset: 'natural',
  sourceFile: null,
  aiResultUrl: '',
  aiBusy: false,
  // ...
});
```

Each UI module imports `state` and `on`, registers listeners for relevant keys, and re-renders only its own DOM subtree when those keys change.

### 5.3 Hardcoded Data Cleanup

- Remove hardcoded `<img src="/data/uploads/img_77f62618a9b5.png">` → replace with placeholder that prompts upload
- Remove `fallback = "/data/outputs/out_de5f8e95e54a.png"` → show upload prompt instead
- `exportButton` href starts empty, set dynamically after upload/AI result
- Delete legacy root-level `index.html`, `app.js`, `styles.css`

---

## 6. API Design Changes

### 6.1 Endpoint Changes

| Method | Path | v0.2.0 | v0.3.0 |
|--------|------|--------|--------|
| POST | `/api/retouch/jobs` | 200 (blocking, ~30s) | **202** (queued, immediate) |
| GET | `/api/retouch/jobs/{id}` | 200 | 200 (add `queued` status) |
| GET | `/api/retouch/jobs` | — | **NEW**: list recent jobs |
| POST | `/api/photos/upload` | 200 | 200 (add auth required) |

### 6.2 New Response Shape

```json
// POST /api/retouch/jobs → 202 Accepted
{
  "jobId": "job_a1b2c3d4e5f6",
  "status": "queued",
  "createdAt": "2026-07-02T10:30:00Z"
}

// GET /api/retouch/jobs/{jobId} → 200 OK (polling)
{
  "jobId": "job_a1b2c3d4e5f6",
  "status": "succeeded",        // queued | running | succeeded | failed
  "sourceImageId": "img_77f62618a9b5",
  "planId": "natural",
  "plan": { ... },
  "modelProvider": "qwen",
  "modelName": "qwen-image-2.0-pro",
  "brainProvider": "qwen",
  "brainModel": "qwen3-vl-plus",
  "outputUrls": ["/data/outputs/out_b2c3d4e5f6a7.png"],
  "errorMessage": null,
  "createdAt": "2026-07-02T10:30:00Z",
  "updatedAt": "2026-07-02T10:30:42Z"
}
```

---

## 7. Data Flow — AI Retouch (v0.3.0)

```
1. Client                              Server
   │  POST /api/photos/upload          │
   │──────────────────────────────────>│ validate file → save → return PhotoAsset
   │  {imageId, url}                   │
   │<──────────────────────────────────│
   │                                    │
   │  POST /api/photos/analyze         │
   │──────────────────────────────────>│ domain analyzer → baseline analysis
   │  {analysis}                       │ brain.analyze() → enriched analysis
   │<──────────────────────────────────│
   │                                    │
   │  POST /api/retouch/plans          │
   │──────────────────────────────────>│ domain planner → 3 RetouchPlan
   │  {plans: [...]}                   │
   │<──────────────────────────────────│
   │                                    │
   │  POST /api/retouch/jobs           │
   │  {plan, instruction}              │
   │──────────────────────────────────>│ create job (status=queued)
   │  202 {jobId, status:"queued"}     │ enqueue background task ──┐
   │<──────────────────────────────────│                             │
   │                                    │                    ┌───────▼────────┐
   │  GET /api/retouch/jobs/{jobId}    │                    │ brain.optimize  │
   │──────────────────────────────────>│                    │ provider.edit   │
   │  {status:"running"}              │                    │ save result     │
   │<──────────────────────────────────│                    │ update status   │
   │  ...poll every 2s...              │                    └────────────────┘
   │                                    │
   │  GET /api/retouch/jobs/{jobId}    │
   │──────────────────────────────────>│
   │  {status:"succeeded",             │
   │   outputUrls:[...]}               │
   │<──────────────────────────────────│
   │                                    │
   │  Fetch /data/outputs/out_xxx.png   │
   │──────────────────────────────────>│ serve static file
   │  {image bytes}                    │
   │<──────────────────────────────────│
```

---

## 8. Directory Structure (v0.3.0)

```
photo-retouch-agent/
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── dependencies.py
│   │   │   ├── model_selection.py
│   │   │   ├── middleware/
│   │   │   │   ├── auth.py            NEW
│   │   │   │   ├── rate_limit.py      NEW
│   │   │   │   └── logging.py         NEW
│   │   │   └── routes/
│   │   │       ├── health.py
│   │   │       ├── photos.py
│   │   │       ├── analyze.py
│   │   │       └── retouch.py
│   │   ├── brains/                    (async method signatures)
│   │   ├── providers/                 (async method signatures)
│   │   ├── domains/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── storage.py             (async)
│   │   │   ├── job_store.py           (SQLite)
│   │   │   ├── retouch_service.py     (async orchestrator)
│   │   │   └── task_queue.py          NEW
│   │   └── core/
│   │       ├── config.py
│   │       ├── errors.py
│   │       ├── database.py            NEW
│   │       └── logging.py             NEW
│   └── tests/                         (async test cases)
├── frontend/
│   ├── index.html                     (no hardcoded src)
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js                     REFACTORED
│       ├── state.js                   NEW
│       ├── api.js
│       ├── ui/
│       │   ├── header.js              NEW
│       │   ├── tool-rail.js           NEW
│       │   ├── category-panel.js      NEW
│       │   ├── canvas-panel.js        NEW
│       │   ├── inspector-panel.js     NEW
│       │   └── ai-panel.js            NEW
│       ├── engine/
│       │   ├── local-retouch.mjs
│       │   └── export-image.mjs
│       └── workflow/
│           └── ai-retouch.mjs
├── docs/
│   └── architecture-v0.3.0.md         THIS FILE
├── pyproject.toml                     (fixed version: >=3.11)
├── .env.example
└── data/                              (runtime data, gitignored)
    ├── uploads/
    ├── outputs/
    ├── jobs/                           REMOVED (→ SQLite)
    └── photo_retouch.db               NEW
```

Removed:
- `index.html` (root) — legacy prototype
- `app.js` (root) — legacy prototype
- `styles.css` (root) — legacy prototype
- `data/jobs/*.json` — replaced by SQLite

---

## 9. Configuration Changes

```bash
# .env.example additions
PHOTO_AGENT_API_KEY=           # NEW: static API key for auth
PHOTO_AGENT_RATE_LIMIT=10      # NEW: requests per second per IP
PHOTO_AGENT_MAX_CONCURRENT=3   # NEW: max concurrent AI jobs
PHOTO_AGENT_DATABASE_PATH=data/photo_retouch.db  # NEW
PHOTO_AGENT_LOG_LEVEL=INFO     # NEW
```

---

## 10. Migration Plan

### Phase 1: Foundation (Week 1)
1. Add `core/database.py` with SQLite schema
2. Add `core/logging.py` with structlog
3. Fix `pyproject.toml` Python version to `>=3.11`
4. Delete legacy root files

### Phase 2: Async Conversion (Week 1-2)
5. Convert all Protocol signatures to async
6. Convert `OpenAICompatiblePromptBrain._complete` to `httpx.AsyncClient`
7. Convert `QwenImageProvider.edit_image` and `SeedreamImageProvider.edit_image`
8. Convert `RetouchService.create_job` and route handlers
9. Update all tests for async

### Phase 3: Task Queue (Week 2)
10. Implement `services/task_queue.py`
11. Change `POST /api/retouch/jobs` to return 202
12. Add polling support in frontend `api.js`
13. Add job listing endpoint

### Phase 4: Frontend Refactor (Week 2-3)
14. Extract `state.js` reactive store
15. Split `main.js` into `ui/*.js` modules
16. Remove hardcoded image paths
17. Add upload prompt placeholder

### Phase 5: Hardening (Week 3)
18. Add `api/middleware/auth.py`
19. Add `api/middleware/rate_limit.py`
20. Add structured logging throughout
21. Replace bare `except Exception` with typed error handling
22. End-to-end integration test

### Phase 6: Polish
23. SSE push for job completion (optional, v0.3.1)
24. Real face detection integration (v0.4.0)

---

## 11. Open Questions

1. **SSE vs Polling for job status**: Polling is simpler and works with any HTTP client. SSE pushes completion instantly but adds complexity. Recommendation: start with polling, add SSE in v0.3.1 if UX demands it.

2. **SQLite vs Postgres**: SQLite is zero-config and sufficient for single-server MVP. If multi-replica deployment is needed, migration to Postgres is straightforward since we use `aiosqlite` with raw SQL (not an ORM).

3. **Background task durability**: In-process `asyncio.create_task` tasks are lost on server restart. For MVP this is acceptable (jobs are short-lived). If durability is needed, switch to a proper queue (Redis + RQ / Celery).

4. **Frontend framework**: Staying with vanilla JS is a deliberate choice for MVP simplicity. If UI complexity grows significantly, consider Preact or Alpine.js — both are lightweight and don't require a build step.
