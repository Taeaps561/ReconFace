"""
ReconFace FastAPI Application Entrypoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.services.face_engine import FaceEngine
from app.services.storage import StorageService
from app.services.vector_db import VectorDBService

settings = get_settings()
setup_logging(level=settings.log_level, is_production=settings.is_production)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to initialize and cleanup resources."""
    logger.info("Starting up %s (env=%s)...", settings.app_name, settings.app_env)

    # Initialize services
    try:
        storage = StorageService()
        storage.initialize()
        logger.info("Storage service initialized (backend: %s)", storage.backend_type)
    except Exception as e:
        logger.warning("Storage initialization notice: %s", e)

    try:
        vector_db = VectorDBService()
        vector_db.ensure_collection()
        logger.info("Vector DB initialized (collection: %s)", vector_db.collection_name)
    except Exception as e:
        logger.warning("Vector DB initialization notice: %s", e)

    # Pre-warm FaceEngine to avoid request-time cold start delay and timeouts
    try:
        engine = FaceEngine.get_instance()
        engine.initialize()
        logger.info("FaceEngine pre-warmed successfully during startup")
    except Exception as e:
        logger.error("FaceEngine pre-warm failure: %s", e)

    yield

    logger.info("Shutting down %s...", settings.app_name)


def create_app() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title="ReconFace API",
        description="Production-grade Reverse Face Recognition & OSINT Search System",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    register_exception_handlers(app)

    # Include Routers
    app.include_router(api_v1_router)

    # Root Web UI
    from pathlib import Path
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    template_path = Path(__file__).parent / "templates" / "index.html"
    dataset_img_path = settings.resolved_dataset_images_dir
    if dataset_img_path.exists():
        app.mount("/dataset-images", StaticFiles(directory=str(dataset_img_path)), name="dataset-images")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui():
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>ReconFace API is running. Go to /docs for Swagger UI</h1>")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
