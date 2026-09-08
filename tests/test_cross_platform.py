"""
Tests for cross-platform path resolution, execution providers, and device detection.
"""

from pathlib import Path
from starlette.testclient import TestClient
from app.core.config import get_settings
from app.services.face_engine import FaceEngine


def test_cross_platform_paths_resolution():
    settings = get_settings()

    # Verify project_root is resolved and is a valid directory
    assert isinstance(settings.project_root, Path)
    assert settings.project_root.exists()

    # Verify dataset paths resolve relative to project root
    assert settings.resolved_dataset_dir == (settings.project_root / "dataset_dusit").resolve()
    assert settings.resolved_dataset_images_dir == (settings.project_root / "dataset_dusit" / "images").resolve()
    assert settings.resolved_dataset_metadata_path == (settings.project_root / "dataset_dusit" / "metadata.json").resolve()
    assert settings.resolved_temp_scan_dir == (settings.project_root / "temp_web_scans").resolve()


def test_face_engine_device_and_providers():
    engine = FaceEngine.get_instance()
    # Check that active_device_name returns a valid non-empty string
    device = engine.active_device_name
    assert isinstance(device, str)
    assert len(device) > 0


def test_health_endpoint_includes_face_engine(api_client: TestClient):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "services" in data
    assert "face_engine" in data["services"]
    face_service = data["services"]["face_engine"]
    assert "status" in face_service
    assert "details" in face_service
    assert "device" in face_service["details"]
    assert "providers" in face_service["details"]
