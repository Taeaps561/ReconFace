"""
Domain models for face detection, embeddings, and quality reporting.

These models are used across the face engine, vector DB service, and API layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for a detected face."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) as integers for array slicing."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class QualityReport(BaseModel):
    """Quality assessment results for a single face crop."""

    model_config = ConfigDict(frozen=True)

    is_sharp: bool = True
    blur_score: float = 0.0
    meets_min_size: bool = True
    face_width: int = 0
    face_height: int = 0
    passed: bool = True

    @property
    def summary(self) -> str:
        issues: list[str] = []
        if not self.is_sharp:
            issues.append(f"blurry (score={self.blur_score:.1f})")
        if not self.meets_min_size:
            issues.append(f"too small ({self.face_width}x{self.face_height})")
        return "OK" if self.passed else "; ".join(issues)


class FaceDetection(BaseModel):
    """A single detected face with its bounding box and confidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    face_index: int = 0
    quality: QualityReport = Field(default_factory=QualityReport)
    landmarks: list[list[float]] | None = None  # 5-point landmarks (optional)


class FaceResult(BaseModel):
    """Complete result for one detected face including its embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection: FaceDetection
    embedding: list[float]  # 512-d normalized vector stored as list for serialization

    @property
    def embedding_array(self) -> np.ndarray:
        """Return embedding as a numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class FaceVector(BaseModel):
    """A face embedding with full metadata, ready for Qdrant upsert."""

    model_config = ConfigDict(frozen=True)

    vector_id: str  # UUID for Qdrant point
    embedding: list[float]  # 512-d
    source_url: str = ""
    face_index: int = 0
    bbox: list[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    thumbnail_path: str = ""
    original_image_path: str = ""
    quality_score: float = 0.0
    indexed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_payload(self) -> dict[str, Any]:
        """Return the metadata payload for Qdrant (excludes the embedding)."""
        return {
            "source_url": self.source_url,
            "face_index": self.face_index,
            "bbox": self.bbox,
            "thumbnail_path": self.thumbnail_path,
            "original_image_path": self.original_image_path,
            "quality_score": self.quality_score,
            "indexed_at": self.indexed_at,
        }
