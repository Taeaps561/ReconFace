"""
Application configuration management using pydantic-settings.

All settings are loaded from environment variables with `.env` file support.
Grouped by service concern for clarity and maintainability.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all ReconFace services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────
    app_name: str = "ReconFace"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Face Engine ────────────────────────────────────────────
    face_model_name: str = "buffalo_l"
    face_det_size: int = 640
    face_ctx_id: int = 0  # 0 = GPU device index, -1 = CPU
    face_min_size: int = 24  # Minimum face bounding box side in pixels (relaxed for web group/avatar photos)
    face_blur_threshold: float = 30.0  # Laplacian variance below this = blurry (relaxed for compressed web images)

    # ── Qdrant ─────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "reconface_vectors"
    qdrant_in_memory: bool = False  # Use in-memory mode for testing

    # ── Redis / Celery ─────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── MinIO / S3 ─────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_originals: str = "recon-originals"
    minio_bucket_faces: str = "recon-faces"

    # ── Storage & Dataset Paths (Cross-Platform) ───────────────
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    dataset_dir: Path | None = None
    temp_dir: Path | None = None
    execution_provider_override: str | None = None  # None = auto, or "CUDA", "CoreML", "DML", "CPU"

    # ── Search Defaults ────────────────────────────────────────
    search_default_top_k: int = Field(default=10, ge=1, le=100)
    search_default_threshold: float = Field(default=0.42, ge=0.0, le=1.0)

    # ── Derived properties ─────────────────────────────────────
    @property
    def resolved_dataset_dir(self) -> Path:
        """Absolute path to the dataset directory."""
        if self.dataset_dir:
            return Path(self.dataset_dir).resolve()
        return (self.project_root / "dataset_dusit").resolve()

    @property
    def resolved_dataset_images_dir(self) -> Path:
        """Absolute path to dataset images directory."""
        return self.resolved_dataset_dir / "images"

    @property
    def resolved_dataset_metadata_path(self) -> Path:
        """Absolute path to dataset metadata.json."""
        return self.resolved_dataset_dir / "metadata.json"

    @property
    def resolved_temp_scan_dir(self) -> Path:
        """Absolute path to temporary scan images directory."""
        if self.temp_dir:
            return Path(self.temp_dir).resolve()
        return (self.project_root / "temp_web_scans").resolve()

    @property
    def face_det_size_tuple(self) -> tuple[int, int]:
        """Return detection size as a tuple for InsightFace."""
        return (self.face_det_size, self.face_det_size)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Uses lru_cache so the .env file is only read once per process lifetime.
    """
    return Settings()
