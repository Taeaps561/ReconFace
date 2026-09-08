"""
Pydantic v2 schemas for API v1 request and response models.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, HttpUrl
from app.models.face import BoundingBox, QualityReport
from app.models.search import SearchMatch


class FaceDetectionOut(BaseModel):
    """Face detection detail returned in search response."""
    face_index: int
    bbox: BoundingBox
    confidence: float
    quality: QualityReport
    landmarks: list[list[float]] | None = None


class SearchResponse(BaseModel):
    """Response returned by the /api/v1/search endpoint."""
    query_face_index: int = Field(..., description="Index of the face selected for vector search")
    total_faces_detected: int = Field(..., description="Total faces detected in the uploaded image")
    detected_faces: list[FaceDetectionOut] = Field(
        default_factory=list,
        description="All detected faces with bounding boxes for multi-face navigation",
    )
    matches: list[SearchMatch] = Field(
        default_factory=list,
        description="Top similar face matches from Vector DB",
    )


class IngestUrlRequest(BaseModel):
    """Single or bulk URL ingestion payload."""
    url: HttpUrl | None = Field(None, description="Single image URL to ingest")
    urls: list[HttpUrl] | None = Field(None, description="Bulk image URLs to ingest")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata payload to attach to indexed faces",
    )


class IngestResponse(BaseModel):
    """Response returned by /api/v1/ingest."""
    status: str
    message: str
    task_id: str | None = None
    batch_id: str | None = None
    total_urls: int = 1


class HealthServiceStatus(BaseModel):
    """Health status of an individual dependency."""
    status: str
    details: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Aggregated health check response."""
    status: str
    app_version: str = "0.1.0"
    services: dict[str, HealthServiceStatus]
