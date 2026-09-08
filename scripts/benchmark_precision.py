import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.face_engine import FaceEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset_dusit"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "metadata.json"
REPORT_FILE = DATASET_DIR / "precision_report.html"

TARGET_IMG_PATH = ROOT_DIR / "test_target.jpg"

def run_precision_benchmark(threshold: float = 0.55):
    print("==================================================================")
    print(f"🎯 RUNNING RECONFACE OFFLINE GPU PRECISION BENCHMARK (Threshold: {threshold})")
    print("==================================================================")

    if not METADATA_FILE.exists():
        print(f"❌ Error: {METADATA_FILE} not found. Please wait for downloader to finish.")
        return

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    print(f"📁 Loaded dataset metadata: {len(metadata)} images")

    # Initialize FaceEngine on RTX GPU
    engine = FaceEngine.get_instance()
    t0 = time.time()
    engine.initialize(ctx_id=0)
    print(f"⚡ FaceEngine initialized on GPU in {time.time() - t0:.2f}s")

    # Extract target embeddings
    if not TARGET_IMG_PATH.exists():
        print(f"❌ Error: Target image {TARGET_IMG_PATH} not found.")
        return

    target_bgr = cv2.imread(str(TARGET_IMG_PATH))
    target_results = engine.extract_embeddings(target_bgr, is_query=True)
    if not target_results:
        print(f"❌ No faces detected in target image {TARGET_IMG_PATH}")
        return

    target_embeddings = [np.array(tr.embedding, dtype=np.float32) for tr in target_results]
    for i in range(len(target_embeddings)):
        n = np.linalg.norm(target_embeddings[i])
        if n > 0:
            target_embeddings[i] /= n

    target_matrix = np.stack(target_embeddings)
    centroid_emb = np.mean(target_matrix, axis=0)
    c_n = np.linalg.norm(centroid_emb)
    if c_n > 0:
        centroid_emb /= c_n

    print(f"👤 Target face embeddings extracted: {len(target_embeddings)} vector(s)")

    # Run Benchmark
    total_faces = 0
    matches = []
    candidates = []
    t_start = time.time()

    for idx, item in enumerate(metadata, 1):
        img_path = Path(item["file_path"])
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None or img_bgr.size == 0:
            continue

        h, w = img_bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        face_results = engine.extract_embeddings(img_bgr, is_query=False)
        total_faces += len(face_results)

        for fr in face_results:
            c_emb = np.array(fr.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_emb)
            if c_norm > 0:
                c_emb /= c_norm

            max_sim = float(max(np.dot(t_emb, c_emb) for t_emb in target_embeddings))
            if len(target_embeddings) > 1:
                c_sim = float(np.dot(centroid_emb, c_emb))
                sim = 0.65 * max_sim + 0.35 * c_sim
            else:
                sim = max_sim

            # Save crop for report
            bbox = fr.detection.bbox
            crop = engine.crop_face(img_bgr, bbox)
            crop_rel_path = ""
            if crop is not None and crop.size > 0:
                crop_dir = DATASET_DIR / "crops"
                crop_dir.mkdir(exist_ok=True)
                crop_name = f"crop_{item['id']}_{fr.detection.face_index}.jpg"
                cv2.imwrite(str(crop_dir / crop_name), crop)
                crop_rel_path = f"crops/{crop_name}"

            match_data = {
                "id": item["id"],
                "filename": item["filename"],
                "score": round(sim, 4),
                "crop_rel": crop_rel_path,
                "img_rel": f"images/{item['filename']}",
                "page_title": item.get("page_title", ""),
                "page_url": item.get("page_url", ""),
                "blur_score": round(fr.detection.quality.blur_score, 1),
                "confidence": round(fr.detection.confidence, 3),
            }

            candidates.append(match_data)
            if sim >= threshold:
                matches.append(match_data)

        if idx % 50 == 0 or idx == len(metadata):
            elapsed = time.time() - t_start
            fps = idx / elapsed if elapsed > 0 else 0
            print(f" -> Scanned {idx}/{len(metadata)} images ({fps:.1f} imgs/sec) | Total faces: {total_faces} | Matches: {len(matches)}", flush=True)

    elapsed_total = time.time() - t_start
    candidates.sort(key=lambda x: x["score"], reverse=True)
    matches.sort(key=lambda x: x["score"], reverse=True)

    print("\n==================================================================")
    print("📊 BENCHMARK SUMMARY RESULTS")
    print("==================================================================")
    print(f"⏱️ Total Scan Time: {elapsed_total:.2f}s ({len(metadata)/elapsed_total:.1f} images/sec on RTX GPU)")
    print(f"🖼️ Images Scanned: {len(metadata)}")
    print(f"👤 Total Faces Detected: {total_faces}")
    print(f"🎯 Target Matches Found (>= {threshold}): {len(matches)}")
    print("------------------------------------------------------------------")
    print("🏆 Top 10 Highest Similarity Matches:")
    for rank, m in enumerate(candidates[:10], 1):
        status = "✅ MATCH" if m["score"] >= threshold else "❌ BELOW THRESH"
        print(f"  {rank:2d}. {m['score']*100:5.1f}% | {status} | Quality Blur: {m['blur_score']} | {m['page_title'][:30]}")

    # Generate HTML Visual Report
    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>ReconFace Precision Benchmark Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 24px; }}
        .header {{ display: flex; justify-content: space-between; border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }}
        .card {{ background: #131c2e; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; text-align: center; }}
        .card-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
        .item {{ background: #131c2e; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; padding: 12px; }}
        .item.matched {{ border-color: #10b981; box-shadow: 0 0 12px rgba(16, 185, 129, 0.2); }}
        .thumbs {{ display: flex; gap: 10px; margin-bottom: 10px; }}
        .thumb-crop {{ width: 70px; height: 70px; border-radius: 6px; border: 2px solid #38bdf8; object-fit: cover; }}
        .thumb-full {{ width: calc(100% - 80px); height: 70px; border-radius: 6px; object-fit: cover; }}
        .score {{ font-size: 16px; font-weight: bold; color: #10b981; }}
        .meta {{ font-size: 11px; color: #94a3b8; margin-top: 6px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 ReconFace Offline Precision Benchmark Report</h1>
        <div>Threshold: <strong>{threshold}</strong> | GPU: <strong>NVIDIA RTX 3060 DirectML</strong></div>
    </div>
    <div class="stats">
        <div class="card"><div class="card-val">{len(metadata)}</div><div>Images Analyzed</div></div>
        <div class="card"><div class="card-val">{total_faces}</div><div>Faces Detected</div></div>
        <div class="card"><div class="card-val" style="color: #10b981;">{len(matches)}</div><div>Target Matches</div></div>
        <div class="card"><div class="card-val">{elapsed_total:.1f}s</div><div>Scan Duration</div></div>
    </div>
    <h2>🏆 All Ranked Detections ({len(candidates)} detections):</h2>
    <div class="grid">
"""
    for m in candidates:
        is_m = m["score"] >= threshold
        border_class = "matched" if is_m else ""
        badge_color = "#10b981" if is_m else "#f59e0b"
        html_content += f"""
        <div class="item {border_class}">
            <div class="thumbs">
                <img src="{m['crop_rel']}" class="thumb-crop" alt="Face Crop">
                <img src="{m['img_rel']}" class="thumb-full" alt="Full Photo">
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="score" style="color: {badge_color};">{m['score']*100:.1f}% Match</span>
                <span style="font-size: 11px; color: #94a3b8;">Blur: {m['blur_score']}</span>
            </div>
            <div class="meta" title="{m['page_title']}">{m['page_title'][:35]}...</div>
            <div class="meta"><a href="{m['page_url']}" target="_blank" style="color: #38bdf8; text-decoration: none;">View Article Page &rarr;</a></div>
        </div>
"""
    html_content += """
    </div>
</body>
</html>
"""
    REPORT_FILE.write_text(html_content, encoding="utf-8")
    print(f"\n📑 Visual HTML Report generated: {REPORT_FILE.resolve()}")

if __name__ == "__main__":
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    run_precision_benchmark(thresh)
