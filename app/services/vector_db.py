"""
Qdrant vector database connection manager and search service.

Handles collection lifecycle, HNSW index configuration, face vector
upsert, and similarity search with score thresholding.

Supports both remote Docker-hosted Qdrant and in-memory mode for testing.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.exceptions import VectorDBError
from app.core.logging import get_logger
from app.models.face import FaceVector
from app.models.search import SearchMatch

logger = get_logger("vector_db")

# ── Constants ──────────────────────────────────────────────────
EMBEDDING_DIM = 512
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100


class VectorDBService:
    """
    Qdrant vector database service for face embedding storage and retrieval.

    Manages the connection, collection schema, and provides typed methods
    for upserting face vectors and performing similarity searches.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        in_memory: bool | None = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.qdrant_collection
        self._in_memory = in_memory if in_memory is not None else settings.qdrant_in_memory

        if self._in_memory:
            logger.info("Initializing Qdrant client in-memory mode")
            self._client = QdrantClient(location=":memory:")
        else:
            _host = host or settings.qdrant_host
            _port = port or settings.qdrant_port
            logger.info("Connecting to Qdrant at %s:%d", _host, _port)
            self._client = QdrantClient(host=_host, port=_port)

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection_name

    # ── Collection Management ──────────────────────────────────

    def ensure_collection(self) -> None:
        """
        Create the face vectors collection if it doesn't already exist.

        Uses HNSW index with M=16, ef_construct=100 and Cosine distance
        on 512-dimensional vectors.
        """
        try:
            collections = self._client.get_collections().collections
            existing = [c.name for c in collections]

            if self._collection_name in existing:
                logger.info("Collection '%s' already exists", self._collection_name)
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=HNSW_M,
                    ef_construct=HNSW_EF_CONSTRUCT,
                ),
            )
            logger.info(
                "Created collection '%s' (dim=%d, HNSW m=%d, ef_construct=%d)",
                self._collection_name, EMBEDDING_DIM, HNSW_M, HNSW_EF_CONSTRUCT,
            )
        except Exception as e:
            raise VectorDBError(
                message=f"Failed to ensure collection: {e}",
                details={"collection": self._collection_name},
            ) from e

    def delete_collection(self) -> None:
        """Delete the collection — used for testing teardown."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("Deleted collection '%s'", self._collection_name)
        except Exception as e:
            logger.warning("Failed to delete collection: %s", e)

    # ── Upsert ─────────────────────────────────────────────────

    def upsert_face_vectors(self, faces: list[FaceVector]) -> int:
        """
        Upsert a batch of face vectors with their metadata payloads.

        Args:
            faces: List of FaceVector instances with embeddings and metadata.

        Returns:
            Number of points successfully upserted.
        """
        if not faces:
            return 0

        points = [
            models.PointStruct(
                id=face.vector_id,
                vector=face.embedding,
                payload=face.to_payload(),
            )
            for face in faces
        ]

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.info("Upserted %d face vectors", len(points))
            return len(points)
        except Exception as e:
            raise VectorDBError(
                message=f"Failed to upsert vectors: {e}",
                details={"count": len(faces)},
            ) from e

    # ── Search ─────────────────────────────────────────────────

    def search_similar_faces(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchMatch]:
        """
        Search for faces similar to the query vector.

        Args:
            query_vector: 512-d normalized embedding.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity score (0.0–1.0).

        Returns:
            List of SearchMatch ordered by descending similarity.
        """
        settings = get_settings()
        top_k = top_k or settings.search_default_top_k
        score_threshold = (
            score_threshold if score_threshold is not None
            else settings.search_default_threshold
        )

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )

            matches: list[SearchMatch] = []
            for point in results.points:
                payload: dict[str, Any] = point.payload or {}
                matches.append(
                    SearchMatch(
                        point_id=str(point.id),
                        score=round(float(point.score), 6),
                        source_url=payload.get("source_url", ""),
                        face_index=payload.get("face_index", 0),
                        bbox=payload.get("bbox", []),
                        thumbnail_path=payload.get("thumbnail_path", ""),
                        original_image_path=payload.get("original_image_path", ""),
                        quality_score=payload.get("quality_score", 0.0),
                        indexed_at=payload.get("indexed_at", ""),
                    )
                )

            logger.info(
                "Search returned %d matches (top_k=%d, threshold=%.2f)",
                len(matches), top_k, score_threshold,
            )
            return matches

        except Exception as e:
            raise VectorDBError(
                message=f"Vector search failed: {e}",
                details={"top_k": top_k, "threshold": score_threshold},
            ) from e

    # ── Health & Info ──────────────────────────────────────────

    def get_collection_info(self) -> dict[str, Any]:
        """Return collection statistics for health checks."""
        try:
            info = self._client.get_collection(self._collection_name)
            status_val = "unknown"
            if info.status:
                status_val = info.status.value if hasattr(info.status, "value") else str(info.status)

            vectors_count = getattr(info, "vectors_count", None)
            if vectors_count is None:
                vectors_count = getattr(info, "indexed_vectors_count", getattr(info, "points_count", 0))

            points_count = getattr(info, "points_count", 0)
            segments_count = getattr(info, "segments_count", None)
            if segments_count is None and hasattr(info, "segments"):
                segments_count = len(info.segments)

            return {
                "collection": self._collection_name,
                "status": status_val,
                "vectors_count": vectors_count,
                "points_count": points_count,
                "segments_count": segments_count,
            }
        except UnexpectedResponse:
            return {
                "collection": self._collection_name,
                "status": "not_found",
                "vectors_count": 0,
                "points_count": 0,
            }
        except Exception as e:
            logger.warning("get_collection_info error: %s", e)
            return {
                "collection": self._collection_name,
                "status": "error",
                "error": str(e),
            }

    def is_healthy(self) -> bool:
        """Check if Qdrant is reachable and the collection exists."""
        try:
            info = self.get_collection_info()
            return info.get("status") not in ("error", "not_found")
        except Exception:
            return False

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def generate_vector_id() -> str:
        """Generate a UUID string suitable for Qdrant point IDs."""
        return str(uuid.uuid4())
