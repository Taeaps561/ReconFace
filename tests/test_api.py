"""
Integration and Unit tests for API endpoints (/health, /search, /ingest).
"""

from unittest.mock import MagicMock, patch
import numpy as np
from starlette.testclient import TestClient

from app.models.face import BoundingBox, FaceDetection, FaceResult, QualityReport
from app.models.search import SearchMatch


def test_health_endpoint(api_client: TestClient):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "qdrant" in data["services"]
    assert "storage" in data["services"]


def test_ingest_endpoint_validation_error(api_client: TestClient):
    # Empty body without url or urls
    response = api_client.post("/api/v1/ingest", json={})
    assert response.status_code == 400


@patch("app.api.v1.router.ingest_image_task.delay")
def test_ingest_single_url(mock_delay: MagicMock, api_client: TestClient):
    mock_delay.return_value = MagicMock(id="mock-task-1234")
    response = api_client.post(
        "/api/v1/ingest",
        json={"url": "https://example.com/target_person.jpg", "metadata": {"source": "osint"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["task_id"] == "mock-task-1234"


def test_search_endpoint_with_mocked_face_engine(
    api_client: TestClient, synthetic_face_image_bytes: bytes
):
    from app.api.v1.router import get_face_engine, get_vector_db_service

    mock_embedding = np.random.randn(512).astype(np.float32)
    mock_embedding /= np.linalg.norm(mock_embedding)

    mock_result = FaceResult(
        detection=FaceDetection(
            bbox=BoundingBox(x1=20, y1=20, x2=160, y2=160),
            confidence=0.98,
            face_index=0,
            quality=QualityReport(passed=True, is_sharp=True, meets_min_size=True),
        ),
        embedding=mock_embedding.tolist(),
    )

    mock_match = SearchMatch(
        point_id="00000000-0000-0000-0000-000000000099",
        score=0.945,
        source_url="https://osint-archive.org/target.jpg",
        thumbnail_path="faces/thumb99.jpg",
    )

    mock_engine = MagicMock()
    mock_engine.process_image.return_value = [mock_result]

    mock_vdb = MagicMock()
    mock_vdb.search_similar_faces.return_value = [mock_match]

    api_client.app.dependency_overrides[get_face_engine] = lambda: mock_engine
    api_client.app.dependency_overrides[get_vector_db_service] = lambda: mock_vdb

    response = api_client.post(
        "/api/v1/search?top_k=5&score_threshold=0.6",
        files={"file": ("target.jpg", synthetic_face_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_faces_detected"] == 1
    assert data["query_face_index"] == 0
    assert len(data["matches"]) == 1
    assert data["matches"][0]["point_id"] == "00000000-0000-0000-0000-000000000099"
    assert data["matches"][0]["score"] == 0.945

