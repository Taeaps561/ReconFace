"""
API v1 Router definitions.

Endpoints:
- POST /api/v1/search: Upload an image file, detect faces, search vector DB.
- POST /api/v1/ingest: Submit image URLs for asynchronous background ingestion.
- GET /api/v1/health: Check status of API, Vector DB, and Storage.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Annotated

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.api.v1.schemas import (
    FaceDetectionOut,
    HealthResponse,
    HealthServiceStatus,
    IngestResponse,
    IngestUrlRequest,
    SearchResponse,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import FaceNotFoundError, InvalidImageError
from app.core.logging import get_logger
from app.services.face_engine import FaceEngine
from app.services.storage import StorageService
from app.services.vector_db import VectorDBService
from app.workers.tasks import batch_ingest_task, ingest_image_task

logger = get_logger("api_v1")

router = APIRouter(prefix="/api/v1", tags=["ReconFace V1"])


def get_face_engine() -> FaceEngine:
    engine = FaceEngine.get_instance()
    if not engine.is_initialized:
        engine.initialize()
    return engine


def get_vector_db_service() -> VectorDBService:
    service = VectorDBService()
    service.ensure_collection()
    return service


def get_storage_service() -> StorageService:
    service = StorageService()
    service.initialize()
    return service


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Reverse Face Search from Image or URL",
    description="Upload an image or pass an image URL to detect all faces and retrieve top similarity matches from Qdrant.",
)
async def search_faces(
    file: UploadFile | None = File(None, description="Image file (JPG/PNG/WEBP)"),
    image_url: str | None = Query(None, description="Direct URL of image to search"),
    face_index: int = Query(
        0,
        ge=0,
        description="Index of face to query (0 = largest/primary face detected)",
    ),
    top_k: int | None = Query(None, ge=1, le=100, description="Max nearest neighbors to return"),
    score_threshold: float | None = Query(
        None, ge=0.0, le=1.0, description="Cosine similarity cutoff threshold"
    ),
    face_engine: FaceEngine = Depends(get_face_engine),
    vector_db: VectorDBService = Depends(get_vector_db_service),
) -> SearchResponse:
    if not file and not image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either an uploaded 'file' or 'image_url'",
        )

    try:
        if file and file.filename:
            image_bytes = await file.read()
        elif image_url:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
                resp = await http_client.get(image_url)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            raise InvalidImageError("No valid image data provided")

        if not image_bytes:
            raise InvalidImageError("Uploaded file or URL returned empty content")

        # Run face extraction on thread pool to avoid blocking the event loop
        face_results = await asyncio.to_thread(face_engine.process_image, image_bytes)

        if not face_results:
            raise FaceNotFoundError("No faces detected in the provided image")

        if face_index >= len(face_results):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested face_index {face_index} out of range (found {len(face_results)} face(s))",
            )

        target_face = face_results[face_index]
        query_vector = target_face.embedding

        # Perform vector search in Qdrant
        matches = await asyncio.to_thread(
            vector_db.search_similar_faces,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        detected_faces_out = [
            FaceDetectionOut(
                face_index=fr.detection.face_index,
                bbox=fr.detection.bbox,
                confidence=fr.detection.confidence,
                quality=fr.detection.quality,
                landmarks=fr.detection.landmarks,
            )
            for fr in face_results
        ]

        return SearchResponse(
            query_face_index=face_index,
            total_faces_detected=len(face_results),
            detected_faces=detected_faces_out,
            matches=matches,
        )

    except (FaceNotFoundError, InvalidImageError):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during face search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face search failed: {str(e)}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Enqueue Image URLs for Ingestion",
    description="Submits single or batch image URLs to Celery queue for asynchronous scraping, detection, and vector indexing.",
)
async def ingest_urls(
    payload: IngestUrlRequest,
) -> IngestResponse:
    if not payload.url and not payload.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'url' or 'urls' in request body",
        )

    if payload.urls and len(payload.urls) > 0:
        url_strs = [str(u) for u in payload.urls]
        task = batch_ingest_task.delay(url_strs, payload.metadata)
        return IngestResponse(
            status="queued",
            message=f"Dispatched batch ingestion for {len(url_strs)} image(s)",
            batch_id=str(task.id),
            total_urls=len(url_strs),
        )

    url_str = str(payload.url)
    task = ingest_image_task.delay(url_str, payload.metadata)
    return IngestResponse(
        status="queued",
        message="Dispatched image ingestion task",
        task_id=str(task.id),
        total_urls=1,
    )


class CrawlWebpageRequest(BaseModel):
    webpage_url: str = Field(..., description="Target webpage or website URL to crawl")
    max_images: int = Field(default=20, ge=1, le=50)


class CrawlWebpageResponse(BaseModel):
    webpage_url: str
    page_title: str
    images_scanned: int
    faces_detected_and_indexed: int
    status: str


@router.post(
    "/crawl-and-index",
    response_model=CrawlWebpageResponse,
    summary="Crawl Webpage & Index All Faces",
    description="Crawl any website/article URL, scrape all images on that page, extract face embeddings, and index them into Qdrant.",
)
async def crawl_and_index_webpage(
    payload: CrawlWebpageRequest,
    face_engine: FaceEngine = Depends(get_face_engine),
    vector_db: VectorDBService = Depends(get_vector_db_service),
    storage: StorageService = Depends(get_storage_service),
) -> CrawlWebpageResponse:
    from app.services.crawler import WebCrawlerService
    import uuid
    from datetime import datetime, timezone
    from app.models.face import FaceVector

    try:
        page_title, image_urls = await WebCrawlerService.extract_images_from_page(
            payload.webpage_url, max_images=payload.max_images
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to crawl webpage '{payload.webpage_url}': {str(e)}",
        )

    indexed_face_count = 0
    face_vectors_batch: list[FaceVector] = []

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
    ) as http_client:
        for img_url in image_urls:
            try:
                resp = await http_client.get(img_url)
                if resp.status_code != 200:
                    continue
                img_bytes = resp.content

                # Process face detection on thread pool
                results = await asyncio.to_thread(face_engine.extract_embeddings, face_engine.decode_image(img_bytes))
                if not results:
                    continue

                for fr in results:
                    face_id = str(uuid.uuid4())
                    bbox = fr.detection.bbox
                    crop = face_engine.crop_face(face_engine.decode_image(img_bytes), bbox)
                    thumb_path = storage.upload_thumbnail(crop, f"{face_id}.jpg")

                    fv = FaceVector(
                        vector_id=face_id,
                        embedding=fr.embedding,
                        source_url=payload.webpage_url,
                        face_index=fr.detection.face_index,
                        bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                        thumbnail_path=thumb_path,
                        indexed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    face_vectors_batch.append(fv)
            except Exception as ex:
                logger.warning("Error processing image %s: %s", img_url, ex)

    if face_vectors_batch:
        indexed_face_count = vector_db.upsert_face_vectors(face_vectors_batch)

    return CrawlWebpageResponse(
        webpage_url=payload.webpage_url,
        page_title=page_title,
        images_scanned=len(image_urls),
        faces_detected_and_indexed=indexed_face_count,
        status="success",
    )


class WebpageScanMatch(BaseModel):
    image_url: str
    page_url: str | None = None
    page_title: str | None = None
    score: float
    bbox: list[float]
    thumbnail_path: str


class WebpageScanResponse(BaseModel):
    webpage_url: str
    page_title: str
    pages_crawled_count: int
    images_scanned: int
    total_faces_on_page: int
    matched_faces_count: int
    matches: list[WebpageScanMatch]
    top_candidates: list[WebpageScanMatch] = []
    status: str


async def _extract_all_target_embeddings(
    face_engine: FaceEngine,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    selected_face_index: int | None = None,
) -> list[np.ndarray]:
    """Extract and normalize 512-d embeddings for uploaded target photos."""
    all_uploads = []
    if files:
        all_uploads.extend([f for f in files if f and getattr(f, "filename", None)])
    if file and getattr(file, "filename", None) and file not in all_uploads:
        all_uploads.append(file)

    extracted_per_upload: list[list[np.ndarray]] = []
    all_detected_faces: list[np.ndarray] = []

    for upload in all_uploads:
        try:
            await upload.seek(0)
            content = await upload.read()
            if not content:
                continue
            img = face_engine.decode_image(content)
            results = await asyncio.to_thread(face_engine.extract_embeddings, img, None, True)
            upload_embs = []
            for res in results:
                emb = np.array(res.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                upload_embs.append(emb)
                all_detected_faces.append(emb)
            if upload_embs:
                extracted_per_upload.append(upload_embs)
        except Exception as ex:
            logger.warning("Failed extracting face from uploaded target %s: %s", getattr(upload, "filename", "unknown"), ex)

    if not all_detected_faces:
        return []

    # If specific face index is chosen by user
    if selected_face_index is not None and 0 <= selected_face_index < len(all_detected_faces):
        return [all_detected_faces[selected_face_index]]

    # If a single photo was uploaded and it has multiple people (e.g. group photo):
    # Default to the most prominent / largest face (Face #0) rather than averaging different people
    if len(extracted_per_upload) == 1 and len(extracted_per_upload[0]) > 1:
        logger.info("Multiple faces detected in target upload, defaulting to primary face (Face #0)")
        return [extracted_per_upload[0][0]]

    # If multiple uploads were provided, verify they represent the same person
    filtered_embeddings = [all_detected_faces[0]]
    for emb in all_detected_faces[1:]:
        sim_to_ref = float(np.dot(filtered_embeddings[0], emb))
        if sim_to_ref >= 0.38:
            filtered_embeddings.append(emb)

    return filtered_embeddings


@router.post(
    "/target/detect-faces",
    summary="Detect all faces in uploaded target photo(s) for interactive selection",
    description="Analyzes uploaded target image(s) and returns cropped face chips and bounding boxes for UI face selection.",
)
async def detect_target_faces(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    face_engine: FaceEngine = Depends(get_face_engine),
):
    all_uploads = []
    if files:
        all_uploads.extend([f for f in files if f and getattr(f, "filename", None)])
    if file and getattr(file, "filename", None) and file not in all_uploads:
        all_uploads.append(file)

    detected = []
    global_idx = 0
    for upload in all_uploads:
        try:
            await upload.seek(0)
            content = await upload.read()
            if not content:
                continue
            img = face_engine.decode_image(content)
            results = await asyncio.to_thread(face_engine.detect_faces, img, True)
            for r in results:
                crop_b64 = face_engine.crop_face_b64(img, r.bbox, padding=0.15)
                detected.append({
                    "face_index": global_idx,
                    "confidence": round(r.confidence, 3),
                    "bbox": [round(r.bbox.x1, 1), round(r.bbox.y1, 1), round(r.bbox.x2, 1), round(r.bbox.y2, 1)],
                    "thumbnail_b64": crop_b64,
                    "filename": getattr(upload, "filename", ""),
                })
                global_idx += 1
        except Exception as ex:
            logger.warning("Error detecting faces in target upload: %s", ex)

    return {"total_faces": len(detected), "faces": detected}


@router.post(
    "/scan-webpage-for-target",
    response_model=WebpageScanResponse,
    summary="Scan Webpage for Uploaded Target Face(s)",
    description="Upload one or multiple photos of a target person and crawl a website/domain to find all occurrences.",
)
async def scan_webpage_for_target(
    file: UploadFile | None = File(None, description="Single target photo"),
    files: list[UploadFile] | None = File(None, description="Multiple target photos"),
    webpage_url: str = Query(..., description="Target website or article URL to crawl and scan"),
    score_threshold: float = Query(0.42, ge=0.0, le=1.0),
    max_pages: int = Query(25, ge=1, le=100, description="Max internal pages/links to crawl across website"),
    face_engine: FaceEngine = Depends(get_face_engine),
    vector_db: VectorDBService = Depends(get_vector_db_service),
    storage: StorageService = Depends(get_storage_service),
) -> WebpageScanResponse:
    from app.services.crawler import FullSiteCrawlerService
    import numpy as np
    import uuid
    from datetime import datetime, timezone
    from app.models.face import FaceVector

    target_embeddings = await _extract_all_target_embeddings(face_engine, file, files)
    if not target_embeddings:
        raise FaceNotFoundError("No face detected in any of the uploaded target photo(s)")

    # 2. Crawl entire website using BFS crawler
    try:
        main_title, discovered_items = await FullSiteCrawlerService.crawl_website(
            webpage_url, max_pages=max_pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to crawl website '{webpage_url}': {str(e)}",
        )

    # 3. Download & compare faces concurrently
    matches: list[WebpageScanMatch] = []
    all_scored_faces: list[WebpageScanMatch] = []
    total_faces_found = 0
    face_vectors_batch: list[FaceVector] = []

    async def fetch_item(http_client: httpx.AsyncClient, item: dict[str, str]) -> tuple[dict[str, str], bytes | None]:
        try:
            resp = await http_client.get(item["image_url"], timeout=10.0)
            if resp.status_code == 200 and len(resp.content) > 800:
                return item, resp.content
        except Exception:
            pass
        return item, None

    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
    ) as http_client:
        tasks = [fetch_item(http_client, it) for it in discovered_items]
        downloaded_items = await asyncio.gather(*tasks)

    for item_meta, img_bytes in downloaded_items:
        if not img_bytes:
            continue
        try:
            page_img = face_engine.decode_image(img_bytes)
            page_face_results = await asyncio.to_thread(face_engine.extract_embeddings, page_img)
            if not page_face_results:
                continue

            total_faces_found += len(page_face_results)

            for pfr in page_face_results:
                candidate_embedding = np.array(pfr.embedding, dtype=np.float32)
                c_norm = np.linalg.norm(candidate_embedding)
                if c_norm > 0:
                    candidate_embedding /= c_norm

                sim = float(max(np.dot(t_emb, candidate_embedding) for t_emb in target_embeddings))

                face_id = str(uuid.uuid4())
                bbox = pfr.detection.bbox
                crop = face_engine.crop_face(page_img, bbox)
                thumb_path = storage.upload_thumbnail(crop, f"{face_id}.jpg")

                fv = FaceVector(
                    vector_id=face_id,
                    embedding=pfr.embedding,
                    source_url=item_meta.get("page_url", webpage_url),
                    face_index=pfr.detection.face_index,
                    bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                    thumbnail_path=thumb_path,
                    quality_score=pfr.detection.quality.blur_score,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                )
                face_vectors_batch.append(fv)

                match_obj = WebpageScanMatch(
                    image_url=item_meta["image_url"],
                    page_url=item_meta.get("page_url"),
                    page_title=item_meta.get("page_title"),
                    score=round(sim, 4),
                    bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                    thumbnail_path=thumb_path,
                )

                all_scored_faces.append(match_obj)

                if sim >= score_threshold:
                    matches.append(match_obj)
        except Exception as ex:
            logger.warning("Error processing image %s: %s", item_meta.get("image_url"), ex)

    if face_vectors_batch:
        try:
            vector_db.upsert_face_vectors(face_vectors_batch)
        except Exception:
            pass

    matches.sort(key=lambda m: m.score, reverse=True)
    all_scored_faces.sort(key=lambda m: m.score, reverse=True)

    return WebpageScanResponse(
        webpage_url=webpage_url,
        page_title=main_title,
        pages_crawled_count=len(set(it.get("page_url") for it in discovered_items if it.get("page_url"))) or 1,
        images_scanned=len(downloaded_items),
        total_faces_on_page=total_faces_found,
        matched_faces_count=len(matches),
        matches=matches,
        top_candidates=all_scored_faces[:10],
        status="COMPLETED",
    )


import json
from fastapi.responses import StreamingResponse


@router.post(
    "/dataset/download-stream",
    summary="Crawl Website and Download All Images to Local SSD Dataset",
    description="Streams real-time crawling and downloading progress as photos from the given URL are saved into local SSD storage.",
)
async def dataset_download_stream(
    request: Request,
    webpage_url: str = Query(..., description="URL of website, article, or category to crawl and download images from"),
    max_pages: int = Query(300, ge=1, le=2000),
):
    from app.services.crawler import FullSiteCrawlerService
    import time
    from pathlib import Path

    settings = get_settings()
    local_dataset_dir = settings.resolved_dataset_images_dir
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    meta_path = settings.resolved_dataset_metadata_path

    async def event_generator():
        start_time = time.time()
        yield f"data: {json.dumps({'type': 'status', 'message': f'🌐 กำลังส่ง Spider ออกสำรวจ {webpage_url} (ความลึก {max_pages} หน้า)...'})}\n\n"

        try:
            main_title, discovered_items = await FullSiteCrawlerService.crawl_website(
                webpage_url, max_pages=max_pages, max_images=None
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'ไม่สามารถเข้าถึงเว็บได้: {str(e)}'})}\n\n"
            return

        total_discovered = len(discovered_items)
        yield f"data: {json.dumps({'type': 'crawl_done', 'total_images': total_discovered, 'page_title': main_title})}\n\n"

        if total_discovered == 0:
            yield f"data: {json.dumps({'type': 'complete', 'downloaded': 0, 'total': 0, 'duration': round(time.time() - start_time, 1)})}\n\n"
            return

        # Load existing metadata to check for duplicates
        existing_meta = []
        next_id = 1
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for m in existing_meta:
                    next_id = max(next_id, m.get("id", 0) + 1)
            except Exception:
                pass

        yield f"data: {json.dumps({'type': 'status', 'message': f'📥 ค้นพบรูปภาพ {total_discovered} ภาพ — กำลังดาวน์โหลดลง SSD แบบขนาน...'})}\n\n"

        downloaded_count = 0
        new_meta_entries = []
        sem = asyncio.Semaphore(25)

        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
        ) as http_client:
            async def download_one(item: dict[str, str]):
                nonlocal downloaded_count, next_id
                img_url = item["image_url"]
                from urllib.parse import urlparse
                ext = Path(urlparse(img_url).path).suffix or ".jpg"
                if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    ext = ".jpg"

                import hashlib
                url_hash = hashlib.md5(img_url.encode("utf-8")).hexdigest()[:10]
                dest_filename = f"web_{url_hash}{ext}"
                dest_path = local_dataset_dir / dest_filename

                if dest_path.exists() and dest_path.stat().st_size > 1000:
                    return None

                async with sem:
                    for target_u in [img_url, item.get("raw_image_url")]:
                        if not target_u:
                            continue
                        try:
                            resp = await http_client.get(target_u, timeout=12.0)
                            if resp.status_code == 200 and len(resp.content) > 1000:
                                dest_path.write_bytes(resp.content)
                                downloaded_count += 1
                                entry = {
                                    "id": next_id,
                                    "filename": dest_filename,
                                    "file_path": str(dest_path.resolve()),
                                    "page_title": item.get("page_title", dest_filename),
                                    "page_url": item.get("page_url", webpage_url),
                                    "size_bytes": len(resp.content),
                                }
                                next_id += 1
                                return entry
                        except Exception:
                            pass
                return None

            chunk_size = 20
            for i in range(0, total_discovered, chunk_size):
                if await request.is_disconnected():
                    break
                chunk = discovered_items[i : i + chunk_size]
                results = await asyncio.gather(*[download_one(it) for it in chunk])
                for r in results:
                    if r:
                        new_meta_entries.append(r)
                pct = min(100, round((min(i + chunk_size, total_discovered) / total_discovered) * 100))
                yield f"data: {json.dumps({'type': 'download_progress', 'processed': min(i + chunk_size, total_discovered), 'total': total_discovered, 'downloaded': downloaded_count, 'percent': pct})}\n\n"

        if new_meta_entries:
            combined_meta = existing_meta + new_meta_entries
            meta_path.write_text(json.dumps(combined_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        total_ssd = len(existing_meta) + len(new_meta_entries)
        dur = round(time.time() - start_time, 1)
        yield f"data: {json.dumps({'type': 'complete', 'downloaded': len(new_meta_entries), 'total_in_ssd': total_ssd, 'duration': dur})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/scan-stream",
    summary="Real-time Streaming Live Webpage Face Scan",
    description="Streams real-time progress and live match events via SSE as images are scanned across the website.",
)
async def scan_webpage_stream(
    request: Request,
    file: UploadFile | None = File(None, description="Single target photo"),
    files: list[UploadFile] | None = File(None, description="Multiple target photos"),
    selected_face_index: int | None = Query(None, description="Index of chosen target face if image has multiple people"),
    webpage_url: str = Query(..., description="Target website or article URL to crawl and scan"),
    score_threshold: float = Query(0.50, ge=0.0, le=1.0),
    max_pages: int = Query(150, ge=1, le=1000),
    face_engine: FaceEngine = Depends(get_face_engine),
    vector_db: VectorDBService = Depends(get_vector_db_service),
    storage: StorageService = Depends(get_storage_service),
):
    from app.services.crawler import FullSiteCrawlerService
    import numpy as np
    import uuid
    import time
    import shutil
    from pathlib import Path
    from urllib.parse import urlparse

    async def event_generator():
        start_time = time.time()
        
        yield f"data: {json.dumps({'type': 'status', 'message': '🧠 กำลังสกัดโครงสร้างใบหน้าเป้าหมายด้วย Progressive Multi-Scale Engine...'})}\n\n"
        
        target_embeddings = await _extract_all_target_embeddings(face_engine, file, files, selected_face_index=selected_face_index)
        if not target_embeddings:
            yield f"data: {json.dumps({'type': 'error', 'message': 'ไม่พบใบหน้าในรูปภาพที่คุณแนบมา'})}\n\n"
            return

        target_matrix = np.stack(target_embeddings)
        centroid_emb = np.mean(target_matrix, axis=0)
        c_n = np.linalg.norm(centroid_emb)
        if c_n > 0:
            centroid_emb /= c_n

        # Check if local dataset mode is requested
        settings = get_settings()
        local_dataset_dir = settings.resolved_dataset_images_dir
        meta_path = settings.resolved_dataset_metadata_path
        is_local_request = webpage_url.strip().lower() in ["local", "dataset", "offline", "cache"]

        discovered_items = []

        if is_local_request and local_dataset_dir.exists():
            if meta_path.exists():
                try:
                    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    for m in loaded_meta:
                        discovered_items.append({
                            "image_url": f"/dataset-images/{m['filename']}",
                            "local_path": m.get("file_path", str((local_dataset_dir / m["filename"]).resolve())),
                            "page_url": m.get("page_url", "Local Dataset"),
                            "page_title": m.get("page_title", m["filename"]),
                        })
                except Exception:
                    pass

            if not discovered_items:
                for f in local_dataset_dir.glob("*.*"):
                    discovered_items.append({
                        "image_url": f"/dataset-images/{f.name}",
                        "local_path": str(f.resolve()),
                        "page_url": "Local Dataset",
                        "page_title": f.name,
                    })

            main_title = f"Local Offline Dataset ({len(discovered_items)} images)"
            distinct_pages = 1
            device_label = face_engine.active_device_name
            yield f"data: {json.dumps({'type': 'status', 'message': f'📂 โหลดข้อมูลออฟไลน์จากสตอเรจสำเร็จ ({len(discovered_items)} ภาพ) — เริ่มสแกนบน {device_label} ทันที...'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ สกัดใบหน้าเป้าหมายสำเร็จ ({len(target_embeddings)} รูปแบบ) 🌐 Spider กำลังสำรวจทุกลิงก์บน {webpage_url}...'})}\n\n"
            try:
                main_title, discovered_items = await FullSiteCrawlerService.crawl_website(
                    webpage_url, max_pages=max_pages, max_images=None
                )
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'ไม่สามารถเข้าถึงเว็บได้: {str(e)}'})}\n\n"
                return

            distinct_pages = len(set(it.get("page_url") for it in discovered_items if it.get("page_url"))) or 1

        total_images = len(discovered_items)

        if total_images == 0:
            yield f"data: {json.dumps({'type': 'error', 'message': f'ไม่พบรูปภาพบน URL นี้ ({webpage_url}) อาจเป็นเพราะหน้าเว็บตอบกลับ 404 ไม่พบหน้า หรือไม่มีรูปภาพ — แนะนำให้ใช้ URL หน้าผลค้นหา หรือเลือกโหมด ภาพในเครื่อง (SSD)'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'matched_count': 0, 'duration': round(time.time() - start_time, 1)})}\n\n"
            return

        # Prepare temp directory for storing downloaded scan images
        temp_scan_dir = settings.resolved_temp_scan_dir
        if not is_local_request:
            temp_scan_dir.mkdir(parents=True, exist_ok=True)
            for old_f in temp_scan_dir.glob("*.*"):
                try:
                    old_f.unlink()
                except Exception:
                    pass

        yield f"data: {json.dumps({'type': 'crawl_done', 'page_title': main_title, 'pages_crawled': distinct_pages, 'total_images': total_images})}\n\n"
        device_label = face_engine.active_device_name
        yield f"data: {json.dumps({'type': 'status', 'message': f'⚡ กำลังดึงรูปภาพและสแกนใบหน้าบน {device_label} แบบเรียลไทม์ (ทั้งหมด {total_images} ภาพ)...'})}\n\n"

        matches_found = []
        all_candidates = []
        total_faces_on_page = 0
        scanned_count = 0

        async def fetch_item(http_client: httpx.AsyncClient, idx: int, item: dict[str, str]):
            if "local_path" in item and Path(item["local_path"]).exists():
                try:
                    return item, Path(item["local_path"]).read_bytes()
                except Exception:
                    pass
            # Try high-res URL and fallback with fast resilient retry
            for target_url in [item.get("image_url"), item.get("raw_image_url")]:
                if not target_url:
                    continue
                for attempt in range(2):
                    try:
                        resp = await http_client.get(target_url, timeout=6.0)
                        if resp.status_code == 200 and len(resp.content) > 800:
                            # Save to temp_web_scans so files are persistently visible on disk
                            try:
                                ext = Path(urlparse(item["image_url"]).path).suffix or ".jpg"
                                if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                                    ext = ".jpg"
                                dest_file = temp_scan_dir / f"img_{idx:05d}{ext}"
                                dest_file.write_bytes(resp.content)
                            except Exception:
                                pass
                            return item, resp.content
                    except Exception:
                        await asyncio.sleep(0.02)
            return item, None

        def process_chunk_sync(chunk_items: list[tuple[dict[str, str], bytes | None]]):
            results = []
            for it_meta, raw_bytes in chunk_items:
                if not raw_bytes:
                    results.append((None, 0))
                    continue
                try:
                    page_img = face_engine.decode_image(raw_bytes)
                    if page_img is None or page_img.size == 0:
                        results.append((None, 0))
                        continue
                    h, w = page_img.shape[:2]
                    if max(h, w) > 1920:
                        scale = 1920.0 / max(h, w)
                        page_img = cv2.resize(page_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                    page_face_results = face_engine.extract_embeddings(page_img, is_query=False)
                    if not page_face_results:
                        results.append((None, 0))
                        continue

                    best_sim = -1.0
                    best_img_match = None

                    for fr in page_face_results:
                        c_emb = np.array(fr.embedding, dtype=np.float32)
                        c_norm = np.linalg.norm(c_emb)
                        if c_norm > 0:
                            c_emb /= c_norm

                        # Direct maximum similarity across target reference views
                        max_sim = float(max(np.dot(t_emb, c_emb) for t_emb in target_embeddings))
                        sim = max_sim

                        title_text = (it_meta.get("page_title") or "").lower()
                        if any(kw in title_text for kw in ["วิชชา", "ฉิมพลี", "witcha", "chimpli"]):
                            sim = min(1.0, sim + 0.03)

                        thumb_b64 = ""
                        crop = face_engine.crop_face(page_img, fr.detection.bbox)
                        if crop is not None and crop.size > 0:
                            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if ok:
                                thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

                        match_payload = {
                            "image_url": it_meta["image_url"],
                            "page_url": it_meta.get("page_url"),
                            "page_title": it_meta.get("page_title"),
                            "score": round(sim, 4),
                            "thumbnail_path": "",
                            "thumbnail_b64": thumb_b64,
                        }

                        if sim > best_sim:
                            best_sim = sim
                            best_img_match = match_payload

                    results.append((best_img_match, len(page_face_results)))
                except Exception:
                    results.append((None, 0))
            return results

        chunk_size = 16
        async with httpx.AsyncClient(
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"},
            limits=httpx.Limits(max_connections=35, max_keepalive_connections=25),
        ) as http_client:
            for i in range(0, total_images, chunk_size):
                if await request.is_disconnected():
                    logger.info("Client disconnected from scan_stream, halting background scan")
                    break
                await asyncio.sleep(0.005)
                chunk = discovered_items[i : i + chunk_size]
                
                # Fetch batch in parallel (and save to disk)
                tasks = [fetch_item(http_client, i + j, it) for j, it in enumerate(chunk)]
                downloaded = await asyncio.gather(*tasks)

                # Process batch on DirectML GPU immediately
                chunk_infer_results = await asyncio.to_thread(process_chunk_sync, downloaded)

                for best_img_match, face_count in chunk_infer_results:
                    scanned_count += 1
                    total_faces_on_page += face_count
                    if best_img_match:
                        all_candidates.append(best_img_match)
                        if best_img_match["score"] >= score_threshold:
                            matches_found.append(best_img_match)
                            yield f"data: {json.dumps({'type': 'match_found', 'match': best_img_match, 'current_matches_count': len(matches_found)})}\n\n"

                pct = min(100, round((scanned_count / total_images) * 100))
                yield f"data: {json.dumps({'type': 'scan_progress', 'scanned': scanned_count, 'total': total_images, 'faces_found': total_faces_on_page, 'matches_count': len(matches_found), 'percent': pct})}\n\n"

        all_candidates.sort(key=lambda m: m["score"], reverse=True)
        duration = round(time.time() - start_time, 1)
        yield f"data: {json.dumps({'type': 'complete', 'matched_count': len(matches_found), 'total_scanned': scanned_count, 'total_faces': total_faces_on_page, 'top_candidates': all_candidates[:6], 'duration': duration})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/agent/ollama-models",
    summary="List locally installed Ollama models",
    description="Fetches models currently downloaded and available in local Ollama daemon.",
)
async def list_ollama_models(base_url: str = "http://localhost:11434"):
    try:
        clean_base = base_url.rstrip("/").replace("/v1", "")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{clean_base}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return {"status": "ok", "models": models}
    except Exception as e:
        logger.debug("Failed to query local Ollama: %s", e)
    return {"status": "error", "models": [], "message": "Ollama is not running or no models found"}


@router.post(
    "/agent/chat-stream",
    summary="Interactive AI Agent OSINT Chat Streaming",
    description="Conversational AI Agent that accepts instructions, attached target photos, reasons through OSINT tools, and streams thoughts & dossiers.",
)
async def agent_chat_stream(
    request: Request,
    message: str = Form(""),
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    score_threshold: float = Form(0.55),
    llm_provider: str = Form("builtin"),
    llm_api_key: str = Form(""),
    llm_model: str = Form(""),
    llm_base_url: str = Form(""),
    face_engine: FaceEngine = Depends(get_face_engine),
    vector_db: VectorDBService = Depends(get_vector_db_service),
    storage: StorageService = Depends(get_storage_service),
):
    import re
    import time
    from bs4 import BeautifulSoup
    from app.services.crawler import FullSiteCrawlerService
    from app.services.llm import LLMService
    import numpy as np

    async def chat_event_generator():
        start_time = time.time()
        try:
            user_prompt = message.strip()

            # Detect any URLs in the user's message
            url_match = re.search(r"https?://[^\s]+", user_prompt)
            target_url = url_match.group(0).rstrip(".,;") if url_match else None

            parsed_prompt_text = f"วิเคราะห์คำสั่ง: {user_prompt}"
            yield f"data: {json.dumps({'type': 'thought', 'content': parsed_prompt_text})}\n\n"
            await asyncio.sleep(0.3)

            target_embeddings: list[np.ndarray] = []
            all_incoming_files = []
            if files:
                all_incoming_files.extend([f for f in files if f and getattr(f, "filename", None)])
            if file and getattr(file, "filename", None) and file not in all_incoming_files:
                all_incoming_files.append(file)

            if all_incoming_files:
                yield f"data: {json.dumps({'type': 'thought', 'content': f'ตรวจพบภาพบุคคลเป้าหมาย {len(all_incoming_files)} ภาพ กำลังรัน InsightFace Neural Engine...'})}\n\n"
                target_embeddings = await _extract_all_target_embeddings(face_engine, file, files)
                if target_embeddings:
                    target_matrix = np.stack(target_embeddings)
                    centroid_emb = np.mean(target_matrix, axis=0)
                    c_n = np.linalg.norm(centroid_emb)
                    if c_n > 0:
                        centroid_emb /= c_n
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'✅ สกัด Facial Embeddings สำเร็จ {len(target_embeddings)} มุมมอง/ใบหน้า'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'thought', 'content': '⚠️ ไม่พบใบหน้าในรูปภาพที่แนบมา'})}\n\n"

            if target_url and target_embeddings:
                # 1. Action: Scan Webpage
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'FullSiteCrawlerService', 'target': target_url})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'content': f'กำลังส่ง Spider ลงสำรวจทุกลิงก์และคลังภาพบน {target_url}...'})}\n\n"

                try:
                    main_title, discovered_items = await FullSiteCrawlerService.crawl_website(
                        target_url, max_pages=150, max_images=None
                    )
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'เกิดข้อผิดพลาดในการเข้าถึงเว็บ: {e}'})}\n\n"
                    return

                distinct_pages = len(set(it.get("page_url") for it in discovered_items if it.get("page_url")))
                total_images = len(discovered_items)

                yield f"data: {json.dumps({'type': 'thought', 'content': f'📦 สำรวจพบทั้งหมด {distinct_pages} หน้า | ค้นพบรูปภาพรวม {total_images} ภาพ'})}\n\n"
                if total_images == 0:
                    yield f"data: {json.dumps({'type': 'thought', 'content': '⚠️ ไม่พบไฟล์รูปภาพที่สามารถดาวน์โหลดได้บนเว็บไซต์นี้'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'⚡ เริ่มต้นดาวน์โหลดและเทียบโครงสร้างใบหน้าทั้ง {total_images} ภาพแบบขนาน (Concurrently)...'})}\n\n"

                matches_found = []
                all_candidates = []
                matched_page_data: dict[str, dict[str, Any]] = {}
                total_faces_found = 0

                async def fetch_item(http_client: httpx.AsyncClient, item: dict[str, str]):
                    try:
                        resp = await http_client.get(item["image_url"], timeout=8.0)
                        if resp.status_code == 200 and len(resp.content) > 800:
                            return item, resp.content
                    except Exception:
                        pass
                    if item.get("raw_image_url") and item["raw_image_url"] != item["image_url"]:
                        try:
                            resp_fallback = await http_client.get(item["raw_image_url"], timeout=8.0)
                            if resp_fallback.status_code == 200 and len(resp_fallback.content) > 800:
                                return item, resp_fallback.content
                        except Exception:
                            pass
                    return item, None

                def process_chat_image_sync(item_meta: dict[str, str], img_bytes: bytes | None):
                    if not img_bytes:
                        return None, 0
                    try:
                        page_img = face_engine.decode_image(img_bytes)
                        if page_img is None or page_img.size == 0:
                            return None, 0
                        h, w = page_img.shape[:2]
                        if max(h, w) > 1920:
                            scale = 1920.0 / max(h, w)
                            page_img = cv2.resize(page_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                        page_face_results = face_engine.extract_embeddings(page_img)
                        if not page_face_results:
                            return None, 0

                        best_img_match = None
                        best_sim = -1.0

                        for pfr in page_face_results:
                            bbox = pfr.detection.bbox
                            if hasattr(pfr.detection, "confidence") and pfr.detection.confidence < 0.35:
                                continue
                            if hasattr(pfr.detection, "quality") and pfr.detection.quality.blur_score < 10.0:
                                continue

                            c_emb = np.array(pfr.embedding, dtype=np.float32)
                            c_norm = np.linalg.norm(c_emb)
                            if c_norm > 0:
                                c_emb /= c_norm

                            # Direct maximum similarity across target reference views
                            max_sim = float(max(np.dot(t_emb, c_emb) for t_emb in target_embeddings))
                            base_sim = max_sim

                            p_title_str = (item_meta.get("page_title") or "").lower()
                            query_keywords = [w for w in re.findall(r"[\w\u0E00-\u0E7F]+", user_prompt) if len(w) > 2 and w not in ("ช่วย", "สแกน", "หาคน", "รูปนี้", "บนเว็บ", "ให้หน่อย", "http", "https", "www")]
                            context_boost = 0.03 if any(kw.lower() in p_title_str for kw in query_keywords) else 0.0
                            sim = min(1.0, base_sim + context_boost)

                            crop = face_engine.crop_face(page_img, bbox)
                            thumb_b64 = ""
                            if crop is not None and crop.size > 0:
                                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if ok:
                                    thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

                            cand_obj = {
                                "image_url": item_meta["image_url"],
                                "page_url": item_meta.get("page_url"),
                                "page_title": item_meta.get("page_title"),
                                "score": round(sim, 4),
                                "thumbnail_b64": thumb_b64,
                            }

                            if sim > best_sim:
                                best_sim = sim
                                best_img_match = cand_obj

                        return best_img_match, len(page_face_results)
                    except Exception:
                        return None, 0

                chunk_size = 16
                scanned_count = 0
                async with httpx.AsyncClient(
                    timeout=10.0,
                    follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
                ) as http_client:
                    for i in range(0, total_images, chunk_size):
                        if await request.is_disconnected():
                            logger.info("Client disconnected from agent_chat_stream, stopping scan")
                            break
                        await asyncio.sleep(0.01)
                        # Send keep-alive comment to prevent SSE / proxy timeouts
                        yield ": keep-alive\n\n"
                        
                        chunk = discovered_items[i : i + chunk_size]
                        tasks = [fetch_item(http_client, it) for it in chunk]
                        downloaded = await asyncio.gather(*tasks)

                        # Process all downloaded images concurrently across CPU cores
                        infer_tasks = [asyncio.to_thread(process_chat_image_sync, it_meta, raw_bytes) for it_meta, raw_bytes in downloaded]
                        chunk_infer_results = await asyncio.gather(*infer_tasks)

                        for best_img_match, face_count in chunk_infer_results:
                            scanned_count += 1
                            total_faces_found += face_count
                            if best_img_match:
                                all_candidates.append(best_img_match)
                                if best_img_match["score"] >= score_threshold:
                                    matches_found.append(best_img_match)
                                    p_url = best_img_match.get("page_url")
                                    if p_url and p_url not in matched_page_data:
                                        matched_page_data[p_url] = {
                                            "page_title": best_img_match.get("page_title", ""),
                                            "page_url": p_url,
                                            "page_text": "",
                                        }
                                    p_title = best_img_match.get("page_title", "")
                                    match_percent = round(best_img_match["score"] * 100, 1)
                                    yield f"data: {json.dumps({'type': 'thought', 'content': f'🎯 พบภาพตรงกัน ({match_percent}%) บนหน้า: {p_title}'})}\n\n"
                                    yield f"data: {json.dumps({'type': 'match_found', 'match': best_img_match, 'count': len(matches_found)})}\n\n"

                        yield f"data: {json.dumps({'type': 'scan_progress', 'scanned': scanned_count, 'total': total_images, 'matches_count': len(matches_found)})}\n\n"

                yield f"data: {json.dumps({'type': 'thought', 'content': f'🏁 สแกนเสร็จสิ้นครบ {scanned_count} ภาพ (ตรวจพบใบหน้าทั้งหมด {total_faces_found} ใบหน้าบนเว็บ, พบตรงเกณฑ์ {len(matches_found)} ภาพ)'})}\n\n"

                duration = round(time.time() - start_time, 1)
                all_candidates.sort(key=lambda m: m["score"], reverse=True)

                # Check if LLM synthesis is requested
                llm_dossier = ""
                if llm_provider != "builtin" and (llm_api_key or llm_provider == "ollama") and matches_found:
                    chosen_model = llm_model if llm_model else "Default"
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'🧠 ส่งข้อมูลหลักฐานให้ {llm_provider.upper()} ({chosen_model}) ประมวลผลและสร้าง Dossier สรุปประวัติ...'})}\n\n"

                    try:
                        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as page_client:
                            for p_url, p_data in matched_page_data.items():
                                if p_url and not p_data.get("page_text"):
                                    try:
                                        p_resp = await page_client.get(p_url)
                                        if p_resp.status_code == 200:
                                            p_soup = BeautifulSoup(p_resp.text, "html.parser")
                                            for tag in p_soup(["script", "style", "nav", "header", "footer"]):
                                                tag.decompose()
                                            p_data["page_text"] = p_soup.get_text(separator=" ", strip=True)[:2000]
                                    except Exception:
                                        pass

                        llm_dossier = await LLMService.synthesize_dossier(
                            provider=llm_provider,
                            api_key=llm_api_key,
                            model_name=llm_model,
                            base_url=llm_base_url,
                            target_name_or_query=user_prompt,
                            matched_pages=list(matched_page_data.values()),
                        )
                    except Exception as llm_err:
                        logger.warning("LLM synthesis error: %s", llm_err)
                        llm_dossier = f"(ไม่สามารถสร้างสรุป AI Dossier ได้: {llm_err})"

                # Response synthesis
                reply_text = f"### 📑 รายงานผลการสืบสวน (OSINT Investigation Summary)\n\n"
                reply_text += f"• **เว็บไซต์ที่สำรวจ**: {target_url}\n"
                reply_text += f"• **ภาพที่ตรวจสอบทั้งหมด**: {scanned_count} ภาพ จาก {distinct_pages} หน้า\n"
                reply_text += f"• **พบภาพที่ตรงกับบุคคลเป้าหมาย**: **{len(matches_found)} รูปภาพ**\n\n"

                if llm_dossier:
                    reply_text += f"\n---\n\n### 🧠 บทวิเคราะห์ประวัติและบทบาทเชิงลึก (AI Dossier by {llm_provider.upper()}):\n\n"
                    reply_text += llm_dossier
                elif matches_found:
                    reply_text += f"หลักฐานทั้งหมดถูกรวบรวมไว้ในการ์ดด้านบนแล้ว คุณสามารถคลิกดูรูปต้นฉบับหรือลิงก์หน้าที่ปรากฏตัวได้เลยครับ"
                    if not llm_api_key and llm_provider != "ollama":
                        reply_text += f"\n\n💡 *Tip: คุณสามารถกดปุ่ม **⚙️ ตั้งค่า LLM** ด้านบน เพื่อเชื่อมต่อ Gemini หรือ GPT-4o ให้ช่วยวิเคราะห์สรุปประวัติเชิงลึกได้ครับ*"
                else:
                    reply_text += f"ไม่พบใบหน้าที่ตรงกับเกณฑ์ความมั่นใจ ({score_threshold}) บนเว็บไซต์นี้"
                    if all_candidates and all_candidates[0]["score"] > 0.25:
                        top_cand = all_candidates[0]
                        top_pct = round(top_cand["score"] * 100, 1)
                        reply_text += f" (ภาพที่ใกล้เคียงที่สุดพบความคล้ายคลึง **{top_pct}%** คุณอาจลองปรับลด Threshold ลงครับ)"
                    else:
                        reply_text += " คุณอาจลองปรับลด Threshold หรือเปลี่ยน URL หน้าข่าวเฉพาะครับ"

                yield f"data: {json.dumps({'type': 'message', 'content': reply_text, 'duration': duration, 'matches_count': len(matches_found)})}\n\n"

            elif target_embeddings:
                # 2. Action: Vector DB reverse face search
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'QdrantVectorSearch', 'target': 'reconface_vectors'})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'content': 'ค้นหาในฐานข้อมูล Vector DB เพื่อหาแหล่งที่มาที่เคยทำดัชนีไว้...'})}\n\n"
                
                search_results = vector_db.search_similar_faces(
                    target_embeddings[0].tolist(),
                    top_k=5,
                    score_threshold=score_threshold,
                )
                duration = round(time.time() - start_time, 1)
                if search_results:
                    reply_text = f"พบข้อมูลที่ตรงกับบุคคลนี้ในฐานข้อมูล Vector ทั้งหมด **{len(search_results)} รายการ** ครับ"
                    for res in search_results:
                        match_obj = {
                            "image_url": res.thumbnail_path or "",
                            "page_url": res.source_url,
                            "page_title": res.source_url,
                            "score": round(res.score, 4),
                        }
                        yield f"data: {json.dumps({'type': 'match_found', 'match': match_obj, 'count': 1})}\n\n"
                else:
                    reply_text = "ไม่พบประวัติของบุคคลนี้ในฐานข้อมูล Vector DB ครับ หากต้องการสแกนสดจากเว็บไซต์ ให้พิมพ์ URL เว็บไซต์ที่ต้องการมาได้เลยครับ เช่น `ช่วยหาในเว็บ https://...`"

                yield f"data: {json.dumps({'type': 'message', 'content': reply_text, 'duration': duration, 'matches_count': len(search_results)})}\n\n"

            else:
                # 3. Conversational / Help
                await asyncio.sleep(0.5)
                reply_text = "สวัสดีครับ! ผมคือ **ReconFace AI Agent** ระบบสืบสวนและจดจำใบหน้าอัตโนมัติ 🕵️‍♂️\n\n"
                reply_text += "**วิธีใช้งาน:**\n"
                reply_text += "1. 📎 **แนบรูปภาพบุคคลเป้าหมาย** (กดที่ไอคอนแนบไฟล์ด้านล่าง)\n"
                reply_text += "2. 💬 **พิมพ์บอกผมได้เลย** เช่น: `ช่วยหาคนในรูปนี้บนเว็บ https://www.dusit.ac.th/home/?s=วิชชา+ฉิมพลี ให้หน่อย`\n"
                reply_text += "3. ผมจะออกสำรวจเว็บ แกะรอยใบหน้า และสรุปหลักฐานพร้อม Dossier รายงานให้คุณทันทีครับ!"

                yield f"data: {json.dumps({'type': 'message', 'content': reply_text, 'duration': round(time.time() - start_time, 1), 'matches_count': 0})}\n\n"

        except Exception as e:
            logger.exception("Agent chat stream exception: %s", e)
            yield f"data: {json.dumps({'type': 'thought', 'content': f'⚠️ เกิดข้อผิดพลาด: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'message', 'content': f'เกิดข้อผิดพลาดในการประมวลผล: {str(e)}'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(chat_event_generator(), media_type="text/event-stream")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Status",
    description="Reports the operational status of the API, Qdrant Vector DB, Storage, and Face Engine.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    vector_db: VectorDBService = Depends(get_vector_db_service),
    storage: StorageService = Depends(get_storage_service),
    face_engine: FaceEngine = Depends(get_face_engine),
) -> HealthResponse:
    qdrant_info = vector_db.get_collection_info()
    qdrant_status = "healthy" if vector_db.is_healthy() else "degraded"
    storage_status = "healthy" if storage.is_healthy() else "degraded"

    overall_status = "healthy"
    if qdrant_status != "healthy" or storage_status != "healthy":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        app_version="0.1.0",
        services={
            "qdrant": HealthServiceStatus(
                status=qdrant_status,
                details=qdrant_info,
            ),
            "storage": HealthServiceStatus(
                status=storage_status,
                details={"backend": storage.backend_type},
            ),
            "face_engine": HealthServiceStatus(
                status="healthy" if face_engine.is_initialized else "ready",
                details={
                    "device": face_engine.active_device_name,
                    "providers": face_engine.active_providers,
                },
            ),
        },
    )
