"""
Shared pytest fixtures and utilities for ReconFace test suite.
"""

from __future__ import annotations

import io
from typing import Generator
import cv2
import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.vector_db import VectorDBService


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings override configured for testing."""
    return Settings(
        app_env="development",
        app_debug=True,
        qdrant_in_memory=True,
        qdrant_collection="test_recon_faces",
        minio_secure=False,
    )


@pytest.fixture
def synthetic_face_image_bytes() -> bytes:
    """
    Creates a synthetic 200x200 RGB test image containing sharp geometric facial features
    for blur and quality pre-filtering tests.
    """
    # Create image with high-contrast sharp edges
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (220, 220, 220)  # Light background

    # Face circle
    cv2.circle(img, (100, 100), 70, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (75, 85), 10, (20, 20, 20), -1)
    cv2.circle(img, (125, 85), 10, (20, 20, 20), -1)
    # Mouth
    cv2.ellipse(img, (100, 130), (30, 15), 0, 0, 180, (50, 50, 200), -1)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def blurry_image_bytes() -> bytes:
    """Creates a heavily blurred image to test blur detection filters."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    pil_img = Image.fromarray(blurred)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=20)
    return buf.getvalue()


@pytest.fixture
def in_memory_vector_db() -> Generator[VectorDBService, None, None]:
    """In-memory Qdrant service fixture."""
    vdb = VectorDBService(in_memory=True, collection_name="test_collection")
    vdb.ensure_collection()
    yield vdb
    vdb.delete_collection()


@pytest.fixture
def api_client(in_memory_vector_db: VectorDBService) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with in-memory service dependency overrides."""
    app = create_app()

    from app.api.v1.router import get_vector_db_service, get_storage_service
    from app.services.storage import StorageService

    storage = StorageService()
    storage.initialize()

    app.dependency_overrides[get_vector_db_service] = lambda: in_memory_vector_db
    app.dependency_overrides[get_storage_service] = lambda: storage

    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()

