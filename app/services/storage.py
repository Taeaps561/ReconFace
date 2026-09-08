"""
Object storage service for MinIO / S3-compatible backends.

Handles uploading original images and cropped face thumbnails,
with presigned URL generation for client-side access.

Falls back to local filesystem storage when MinIO is unavailable (dev mode).
"""

from __future__ import annotations

import io
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger("storage")

# Local fallback directory when MinIO is unavailable
LOCAL_STORAGE_DIR = Path("./storage_local")


class StorageService:
    """
    MinIO-backed object storage with local filesystem fallback.

    Auto-creates required buckets on initialization. Provides upload
    and presigned URL generation for both original images and face crops.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
    ) -> None:
        settings = get_settings()
        self._endpoint = endpoint or settings.minio_endpoint
        self._access_key = access_key or settings.minio_access_key
        self._secret_key = secret_key or settings.minio_secret_key
        self._secure = secure if secure is not None else settings.minio_secure
        self._bucket_originals = settings.minio_bucket_originals
        self._bucket_faces = settings.minio_bucket_faces

        self._client: Minio | None = None
        self._use_local: bool = False

    def initialize(self) -> None:
        """Connect to MinIO and ensure buckets exist. Falls back to local FS on failure."""
        import socket
        import urllib3

        # Quick TCP socket check to prevent urllib3 retry delays if MinIO port is closed
        host, _, port_str = self._endpoint.partition(":")
        port = int(port_str) if port_str else (443 if self._secure else 80)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        is_port_open = (sock.connect_ex((host, port)) == 0)
        sock.close()

        if not is_port_open:
            logger.info("MinIO not running on %s, using local file storage at %s", self._endpoint, LOCAL_STORAGE_DIR)
            self._use_local = True
            self._ensure_local_dirs()
            return

        try:
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=1.0, read=2.0),
                retries=urllib3.Retry(total=1, connect=1, read=1),
            )
            self._client = Minio(
                endpoint=self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                http_client=http_client,
            )
            self._client.list_buckets()
            self._ensure_buckets()
            self._use_local = False
            logger.info("MinIO connected at %s", self._endpoint)
        except Exception as e:
            logger.warning(
                "MinIO connection failed (%s), falling back to local storage at %s",
                e, LOCAL_STORAGE_DIR,
            )
            self._use_local = True
            self._ensure_local_dirs()

    def _ensure_buckets(self) -> None:
        """Create required MinIO buckets if they don't exist."""
        if self._client is None:
            return
        for bucket in (self._bucket_originals, self._bucket_faces):
            try:
                if not self._client.bucket_exists(bucket):
                    self._client.make_bucket(bucket)
                    logger.info("Created MinIO bucket: %s", bucket)
            except S3Error as e:
                raise StorageError(
                    message=f"Failed to create bucket '{bucket}': {e}",
                ) from e

    def _ensure_local_dirs(self) -> None:
        """Create local storage directories as a fallback."""
        for subdir in ("originals", "faces"):
            (LOCAL_STORAGE_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ── Upload Methods ─────────────────────────────────────────

    def upload_original(self, image_bytes: bytes, object_name: str) -> str:
        """
        Upload an original image to storage.

        Args:
            image_bytes: Raw image file bytes.
            object_name: Object key / filename (e.g., "abc123.jpg").

        Returns:
            The storage path (MinIO key or local file path).
        """
        if self._use_local:
            return self._save_local("originals", object_name, image_bytes)

        return self._upload_to_minio(
            self._bucket_originals, object_name, image_bytes, "image/jpeg"
        )

    def upload_thumbnail(
        self, face_crop: np.ndarray, object_name: str, quality: int = 90
    ) -> str:
        """
        Encode a face crop as JPEG and upload to storage.

        Args:
            face_crop: BGR numpy array of the cropped face.
            object_name: Object key / filename (e.g., "face_abc123.jpg").
            quality: JPEG compression quality (0–100).

        Returns:
            The storage path.
        """
        success, buffer = cv2.imencode(
            ".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise StorageError(message="Failed to encode face crop as JPEG")

        image_bytes = buffer.tobytes()

        if self._use_local:
            return self._save_local("faces", object_name, image_bytes)

        return self._upload_to_minio(
            self._bucket_faces, object_name, image_bytes, "image/jpeg"
        )

    def _upload_to_minio(
        self, bucket: str, object_name: str, data: bytes, content_type: str
    ) -> str:
        """Upload bytes to a MinIO bucket."""
        if self._client is None:
            raise StorageError(message="MinIO client not initialized")

        try:
            self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            path = f"{bucket}/{object_name}"
            logger.debug("Uploaded to MinIO: %s", path)
            return path
        except S3Error as e:
            raise StorageError(
                message=f"MinIO upload failed: {e}",
                details={"bucket": bucket, "object": object_name},
            ) from e

    def _save_local(self, subdir: str, filename: str, data: bytes) -> str:
        """Save to local filesystem as fallback."""
        path = LOCAL_STORAGE_DIR / subdir / filename
        path.write_bytes(data)
        return str(path)

    # ── URL Generation ─────────────────────────────────────────

    def get_presigned_url(
        self, bucket: str, object_name: str, expires: int = 3600
    ) -> str:
        """
        Generate a presigned download URL for an object.

        Args:
            bucket: Bucket name.
            object_name: Object key.
            expires: URL validity in seconds (default: 1 hour).

        Returns:
            Presigned URL string, or local file path if using fallback.
        """
        if self._use_local or self._client is None:
            return str(LOCAL_STORAGE_DIR / bucket / object_name)

        try:
            return self._client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=timedelta(seconds=expires),
            )
        except S3Error as e:
            raise StorageError(
                message=f"Failed to generate presigned URL: {e}",
            ) from e

    def get_thumbnail_url(self, object_name: str, expires: int = 3600) -> str:
        """Shortcut to get a presigned URL for a face thumbnail."""
        return self.get_presigned_url(self._bucket_faces, object_name, expires)

    def get_original_url(self, object_name: str, expires: int = 3600) -> str:
        """Shortcut to get a presigned URL for an original image."""
        return self.get_presigned_url(self._bucket_originals, object_name, expires)

    # ── Health ─────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Check if storage backend is reachable."""
        if self._use_local:
            return LOCAL_STORAGE_DIR.exists()
        try:
            if self._client:
                self._client.list_buckets()
                return True
        except Exception:
            pass
        return False

    @property
    def backend_type(self) -> str:
        return "local" if self._use_local else "minio"
