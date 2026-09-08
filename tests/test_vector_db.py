"""
Unit tests for Qdrant VectorDBService (HNSW index, Cosine distance, batch upsert, search).
"""

import numpy as np
import pytest
from app.models.face import FaceVector
from app.services.vector_db import VectorDBService


def test_ensure_collection(in_memory_vector_db: VectorDBService):
    info = in_memory_vector_db.get_collection_info()
    assert info["collection"] == "test_collection"
    assert info["status"] != "error"


def test_upsert_and_search_exact_match(in_memory_vector_db: VectorDBService):
    # Generate normalized 512-d random vector
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    vec2 = np.random.randn(512).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)

    face1 = FaceVector(
        vector_id="00000000-0000-0000-0000-000000000001",
        embedding=vec1.tolist(),
        source_url="https://example.com/person1.jpg",
        face_index=0,
        bbox=[10.0, 20.0, 100.0, 120.0],
        thumbnail_path="faces/face1.jpg",
        original_image_path="originals/orig1.jpg",
        quality_score=150.0,
    )

    face2 = FaceVector(
        vector_id="00000000-0000-0000-0000-000000000002",
        embedding=vec2.tolist(),
        source_url="https://example.com/person2.jpg",
        face_index=0,
        bbox=[15.0, 25.0, 110.0, 130.0],
        thumbnail_path="faces/face2.jpg",
        original_image_path="originals/orig2.jpg",
        quality_score=80.0,
    )

    # Upsert batch
    upserted = in_memory_vector_db.upsert_face_vectors([face1, face2])
    assert upserted == 2

    # Query with exact vec1
    matches = in_memory_vector_db.search_similar_faces(
        query_vector=vec1.tolist(),
        top_k=5,
        score_threshold=0.9,
    )

    assert len(matches) >= 1
    top_match = matches[0]
    assert top_match.point_id == face1.vector_id
    assert top_match.source_url == face1.source_url
    assert top_match.score >= 0.99  # Identical cosine distance should be ~1.0


def test_search_score_threshold_filters_out(in_memory_vector_db: VectorDBService):
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    face = FaceVector(
        vector_id="00000000-0000-0000-0000-000000000003",
        embedding=vec.tolist(),
        source_url="https://example.com/person3.jpg",
    )
    in_memory_vector_db.upsert_face_vectors([face])

    # Search with an orthogonal/opposite query and strict threshold
    opposite_vec = (-vec).tolist()
    matches = in_memory_vector_db.search_similar_faces(
        query_vector=opposite_vec,
        top_k=5,
        score_threshold=0.5,
    )
    assert len(matches) == 0
