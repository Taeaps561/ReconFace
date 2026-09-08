"""
Custom exception hierarchy and FastAPI exception handlers.

Provides domain-specific errors that map to consistent JSON error responses
with appropriate HTTP status codes.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ── Base Exception ─────────────────────────────────────────────

class ReconFaceError(Exception):
    """Base exception for all ReconFace domain errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ── Face Engine Errors ─────────────────────────────────────────

class FaceNotFoundError(ReconFaceError):
    """No faces were detected in the provided image."""

    status_code = 422
    error_code = "FACE_NOT_FOUND"

    def __init__(self, message: str = "No faces detected in the provided image") -> None:
        super().__init__(message)


class QualityCheckError(ReconFaceError):
    """Image or face failed quality pre-filtering checks."""

    status_code = 422
    error_code = "QUALITY_CHECK_FAILED"

    def __init__(
        self,
        message: str = "Image failed quality checks",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)


class InvalidImageError(ReconFaceError):
    """The uploaded file could not be decoded as a valid image."""

    status_code = 400
    error_code = "INVALID_IMAGE"

    def __init__(self, message: str = "Unable to decode the provided file as an image") -> None:
        super().__init__(message)


# ── Vector DB Errors ───────────────────────────────────────────

class VectorDBError(ReconFaceError):
    """Error communicating with the Qdrant vector database."""

    status_code = 503
    error_code = "VECTOR_DB_ERROR"


# ── Storage Errors ─────────────────────────────────────────────

class StorageError(ReconFaceError):
    """Error communicating with MinIO / object storage."""

    status_code = 503
    error_code = "STORAGE_ERROR"


# ── Ingestion Errors ──────────────────────────────────────────

class IngestionError(ReconFaceError):
    """Error during the image ingestion pipeline."""

    status_code = 502
    error_code = "INGESTION_ERROR"


# ── FastAPI Exception Handlers ─────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Attach custom exception handlers to the FastAPI application."""

    @app.exception_handler(ReconFaceError)
    async def reconface_error_handler(
        request: Request, exc: ReconFaceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Log the full traceback server-side; return a generic message to clients
        import logging
        logging.getLogger("reconface.api").exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                    "details": {},
                }
            },
        )
