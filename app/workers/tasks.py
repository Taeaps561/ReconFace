"""
Celery tasks for the image ingestion pipeline.

Pipeline per image:
    1. Download image via httpx
    2. Decode & validate (blur, size)
    3. Detect faces → extract 512-d embeddings
    4. Crop face thumbnails → upload to MinIO
    5. Upsert vectors + metadata to Qdrant
    6. Return summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.models.face import FaceVector
from app.services.face_engine import FaceEngine
from app.services.storage import StorageService
from app.services.vector_db import VectorDBService
from app.workers.celery_app import celery_app

logger = get_logger("tasks")


def _get_face_engine() -> FaceEngine:
    """Get or initialize the FaceEngine singleton within the worker process."""
    engine = FaceEngine.get_instance()
    if not engine.is_initialized:
        engine.initialize()
    return engine


def _get_vector_db() -> VectorDBService:
    """Create a VectorDBService instance (connection-pooled by qdrant-client)."""
    svc = VectorDBService()
    svc.ensure_collection()
    return svc


def _get_storage() -> StorageService:
    """Create and initialize a StorageService instance."""
    svc = StorageService()
    svc.initialize()
    return svc


@celery_app.task(
    bind=True,
    name="app.workers.tasks.ingest_image_task",
    max_retries=3,
    default_retry_delay=10,
    rate_limit="30/m",
    acks_late=True,
)
def ingest_image_task(
    self: Any,
    image_url: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Download a single image, extract faces, and index to Qdrant.

    Args:
        image_url: URL of the image to download and process.
        metadata: Optional extra metadata to attach to each face vector.

    Returns:
        Summary dict with counts and vector IDs.
    """
    metadata = metadata or {}
    task_id = self.request.id or str(uuid.uuid4())

    logger.info("Ingesting image: %s (task=%s)", image_url, task_id)

    try:
        # 1. Download image
        image_bytes = _download_image(image_url)

        # 2. Face detection & embedding extraction
        engine = _get_face_engine()
        results = engine.process_image_with_quality_filter(image_bytes, strict=False)

        if not results:
            logger.warning("No faces found in %s", image_url)
            return {
                "task_id": task_id,
                "source_url": image_url,
                "faces_detected": 0,
                "faces_indexed": 0,
                "vector_ids": [],
                "status": "no_faces",
            }

        # 3. Upload original image to storage
        storage = _get_storage()
        original_key = f"{uuid.uuid4().hex}.jpg"
        original_path = storage.upload_original(image_bytes, original_key)

        # 4. Process each face: crop, upload thumbnail, build vector
        vector_db = _get_vector_db()
        face_vectors: list[FaceVector] = []
        image = engine.decode_image(image_bytes)

        for face_result in results:
            face_id = str(uuid.uuid4())
            bbox = face_result.detection.bbox

            # Crop and upload thumbnail
            crop = engine.crop_face(image, bbox)
            thumb_key = f"{face_id}.jpg"
            thumb_path = storage.upload_thumbnail(crop, thumb_key)

            # Build FaceVector for Qdrant
            fv = FaceVector(
                vector_id=face_id,
                embedding=face_result.embedding,
                source_url=image_url,
                face_index=face_result.detection.face_index,
                bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                thumbnail_path=thumb_path,
                original_image_path=original_path,
                quality_score=face_result.detection.quality.blur_score,
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )
            face_vectors.append(fv)

        # 5. Batch upsert to Qdrant
        indexed_count = vector_db.upsert_face_vectors(face_vectors)

        result = {
            "task_id": task_id,
            "source_url": image_url,
            "faces_detected": len(results),
            "faces_indexed": indexed_count,
            "vector_ids": [fv.vector_id for fv in face_vectors],
            "status": "success",
        }
        logger.info("Ingestion complete: %s", result)
        return result

    except Exception as exc:
        logger.exception("Ingestion failed for %s: %s", image_url, exc)
        # Retry on transient errors
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "task_id": task_id,
                "source_url": image_url,
                "faces_detected": 0,
                "faces_indexed": 0,
                "vector_ids": [],
                "status": "failed",
                "error": str(exc),
            }


@celery_app.task(
    name="app.workers.tasks.batch_ingest_task",
    acks_late=True,
)
def batch_ingest_task(
    image_urls: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fan-out ingestion: spawns individual ingest_image_task per URL.

    Uses Celery group for parallel execution across workers.

    Args:
        image_urls: List of image URLs to ingest.
        metadata: Optional metadata to attach to all vectors.

    Returns:
        Summary with task IDs for tracking.
    """
    from celery import group

    metadata = metadata or {}
    tasks = group(
        ingest_image_task.s(url, metadata) for url in image_urls
    )
    result = tasks.apply_async()

    return {
        "batch_id": result.id,
        "total_urls": len(image_urls),
        "task_ids": [r.id for r in result.children] if result.children else [],
        "status": "dispatched",
    }


def _download_image(url: str, timeout: float = 30.0) -> bytes:
    """
    Download an image from a URL using httpx.

    Args:
        url: The image URL.
        timeout: Request timeout in seconds.

    Returns:
        Raw image bytes.

    Raises:
        IngestionError: If the download fails or the response is not an image.
    """
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise IngestionError(
                    message=f"URL did not return an image (content-type: {content_type})",
                    details={"url": url},
                )

            return response.content

    except httpx.HTTPError as e:
        raise IngestionError(
            message=f"Failed to download image from {url}: {e}",
            details={"url": url},
        ) from e
