from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.api.routes.photos import router as photos_router
from app.api.routes.retouch import router as retouch_router
from app.core.config import settings
from app.services.storage import ensure_data_dirs


def create_app() -> FastAPI:
    ensure_data_dirs(settings)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/data", StaticFiles(directory=settings.data_dir), name="data")
    app.include_router(health_router, prefix="/api")
    app.include_router(photos_router, prefix="/api")
    app.include_router(analyze_router, prefix="/api")
    app.include_router(retouch_router, prefix="/api")
    if settings.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
