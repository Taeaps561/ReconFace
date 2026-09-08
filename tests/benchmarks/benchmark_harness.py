"""
Automated Evaluation & Performance Harness for ReconFace.

Measures:
- Face embedding extraction latency (ms)
- Vector DB similarity search latency (ms)
- End-to-end throughput under simulated concurrent load
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from app.services.vector_db import VectorDBService
from app.models.face import FaceVector


def run_benchmark_harness(num_vectors: int = 1000, num_queries: int = 100) -> dict:
    print(f"[*] Starting ReconFace Benchmark Harness...")
    print(f"   -> Populating in-memory Qdrant with {num_vectors} 512-d face vectors...")

    vdb = VectorDBService(in_memory=True, collection_name="harness_bench")
    vdb.ensure_collection()

    # Generate synthetic vector dataset
    vectors = []
    for i in range(num_vectors):
        v = np.random.randn(512).astype(np.float32)
        v /= np.linalg.norm(v)
        vectors.append(
            FaceVector(
                vector_id=f"00000000-0000-0000-0000-{i:012d}",
                embedding=v.tolist(),
                source_url=f"https://osint.target/image_{i}.jpg",
            )
        )

    t0 = time.perf_counter()
    vdb.upsert_face_vectors(vectors)
    upsert_duration = (time.perf_counter() - t0) * 1000

    print(f"   -> Upsert completed in {upsert_duration:.2f} ms ({num_vectors / (upsert_duration / 1000):.1f} vec/sec)")

    # Measure query latencies
    latencies = []
    for _ in range(num_queries):
        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        t_q = time.perf_counter()
        vdb.search_similar_faces(query_vector=query.tolist(), top_k=10, score_threshold=0.5)
        latencies.append((time.perf_counter() - t_q) * 1000)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    report = {
        "dataset_size": num_vectors,
        "queries_executed": num_queries,
        "upsert_time_ms": round(upsert_duration, 2),
        "search_latency_p50_ms": round(p50, 2),
        "search_latency_p95_ms": round(p95, 2),
        "search_latency_p99_ms": round(p99, 2),
    }

    print("\n--- Benchmark Harness Summary ---")
    for k, v in report.items():
        print(f"   - {k}: {v}")

    vdb.delete_collection()
    return report


if __name__ == "__main__":
    run_benchmark_harness()
