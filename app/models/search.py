"""
Domain models for search results and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchMatch(BaseModel):
    """A single match from the vector similarity search."""

    model_config = ConfigDict(frozen=True)

    point_id: str
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")
    source_url: str = ""
    face_index: int = 0
    bbox: list[float] = Field(default_factory=list)
    thumbnail_path: str = ""
    original_image_path: str = ""
    quality_score: float = 0.0
    indexed_at: str = ""


class SearchResult(BaseModel):
    """Aggregated search result for a single query face."""

    query_face_index: int = 0
    query_bbox: list[float] = Field(default_factory=list)
    matches: list[SearchMatch] = Field(default_factory=list)
    total_matches: int = 0
