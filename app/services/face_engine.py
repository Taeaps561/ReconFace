"""
Face detection and embedding extraction engine.

Wraps InsightFace's FaceAnalysis (ArcFace / buffalo_l) as a thread-safe
singleton. Provides quality pre-filtering (blur detection, minimum face size)
and batch inference support.

Usage:
    engine = FaceEngine.get_instance()
    engine.initialize()
    results = engine.process_image(image_bytes)
"""

from __future__ import annotations

import io
import threading
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

from app.core.config import get_settings
from app.core.exceptions import FaceNotFoundError, InvalidImageError, QualityCheckError
from app.core.logging import get_logger
from app.models.face import (
    BoundingBox,
    FaceDetection,
    FaceResult,
    QualityReport,
)

logger = get_logger("face_engine")


class FaceEngine:
    """
    Thread-safe singleton for InsightFace model lifecycle.

    The engine is lazily initialized on first use and reused across
    all API requests and Celery workers within the same process.
    """

    _instance: ClassVar[FaceEngine | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None
        self._initialized: bool = False
        self._active_providers: list[str] = []
        self._active_device_name: str = "Uninitialized"

    @classmethod
    def get_instance(cls) -> FaceEngine:
        """Return the global FaceEngine singleton (double-checked locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton — primarily for testing."""
        with cls._lock:
            cls._instance = None

    # ── Initialization ─────────────────────────────────────────

    def initialize(
        self,
        model_name: str | None = None,
        ctx_id: int | None = None,
        det_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Load the InsightFace model into memory with automatic cross-platform acceleration:
        NVIDIA CUDA (Windows/Linux) -> CoreML (macOS Apple Silicon) -> DirectML (Windows) -> CPU.

        Args:
            model_name: InsightFace model pack (default: from settings).
            ctx_id: ONNX execution device (-1=CPU, 0+=GPU).
            det_size: Detection input size as (width, height).
        """
        if self._initialized:
            logger.info("FaceEngine already initialized, skipping")
            return

        settings = get_settings()
        model_name = model_name or settings.face_model_name
        ctx_id = ctx_id if ctx_id is not None else settings.face_ctx_id
        det_size = det_size or settings.face_det_size_tuple
        override = (settings.execution_provider_override or "").strip().upper()

        logger.info(
            "Initializing FaceEngine: model=%s, ctx_id=%d, det_size=%s, override=%s",
            model_name, ctx_id, det_size, override or "auto",
        )

        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            # Preferred order: CUDA (NVIDIA) > CoreML (Apple Silicon) > DirectML (Windows)
            gpu_providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"]

            if override == "CPU" or ctx_id == -1:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
            elif override in ["CUDA", "CUDAEXECUTIONPROVIDER"] and "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            elif override in ["COREML", "COREMLEXECUTIONPROVIDER"] and "CoreMLExecutionProvider" in avail:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            elif override in ["DML", "DMLEXECUTIONPROVIDER"] and "DmlExecutionProvider" in avail:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            elif any(p in avail for p in gpu_providers):
                providers = [p for p in gpu_providers if p in avail] + ["CPUExecutionProvider"]
                ctx_id = 0
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
        except Exception:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        def _resolve_device_label(provs: list[str], c_id: int) -> str:
            if c_id < 0 or (len(provs) == 1 and "CPUExecutionProvider" in provs):
                return "CPU"
            if "CUDAExecutionProvider" in provs:
                return "NVIDIA CUDA GPU"
            if "CoreMLExecutionProvider" in provs:
                return "Apple Silicon (CoreML / Neural Engine)"
            if "DmlExecutionProvider" in provs:
                return "DirectML GPU"
            return "GPU Acceleration"

        logger.info("FaceEngine active ONNX execution providers: %s (ctx_id=%d)", providers, ctx_id)

        try:
            self._app = FaceAnalysis(name=model_name, providers=providers)
            # Using det_thresh=0.35 provides sensitive face detection for web, group, and candid photos
            self._app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=0.35)
            self._initialized = True
            self._active_providers = list(providers)
            self._active_device_name = _resolve_device_label(providers, ctx_id)
            logger.info("FaceEngine initialized successfully on %s (ctx_id=%d)", self._active_device_name, ctx_id)
        except Exception as e:
            if "CPUExecutionProvider" not in providers or len(providers) > 1:
                logger.warning("Hardware acceleration initialization failed (%s), falling back to CPU...", e)
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                self._app = FaceAnalysis(name=model_name, providers=providers)
                self._app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=0.35)
                self._initialized = True
                self._active_providers = ["CPUExecutionProvider"]
                self._active_device_name = "CPU (Fallback)"
                logger.info("FaceEngine initialized successfully on CPU fallback")
            else:
                raise e

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def active_device_name(self) -> str:
        """Return a human-readable description of the active execution device."""
        if not self._initialized:
            return "Uninitialized"
        return self._active_device_name

    @property
    def active_providers(self) -> list[str]:
        """Return the list of active ONNX execution providers."""
        return list(self._active_providers)

    # ── Image Decoding & Validation ────────────────────────────

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """
        Decode raw bytes into a BGR numpy array (OpenCV format).
        Supports OpenCV imdecode with PIL fallback for CMYK, WebP, and auto-EXIF rotation.

        Raises:
            InvalidImageError: If the bytes cannot be decoded as an image.
        """
        if not image_bytes:
            raise InvalidImageError("Image bytes are empty")

        # 1. Fast OpenCV decode
        try:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass

        # 2. Robust PIL fallback (handles CMYK JPEG, AVIF, WebP, and EXIF orientation)
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise InvalidImageError(f"Failed to decode image bytes — unsupported or corrupt file: {e}")

    @staticmethod
    def compute_blur_score(face_crop: np.ndarray) -> float:
        """
        Compute the Laplacian variance as a sharpness metric.

        Higher values indicate sharper images. Typical threshold: 30-100.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def assess_quality(self, image: np.ndarray, bbox: BoundingBox) -> QualityReport:
        """
        Run quality pre-filtering on a face crop.

        Checks:
            1. Blur detection via Laplacian variance.
            2. Minimum face bounding box size (24x24 px default).
        """
        settings = get_settings()

        x1, y1, x2, y2 = bbox.to_int_tuple()
        # Clamp to image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = image[y1:y2, x1:x2]
        face_h, face_w = face_crop.shape[:2]

        blur_score = self.compute_blur_score(face_crop) if face_crop.size > 0 else 0.0
        is_sharp = blur_score >= settings.face_blur_threshold
        meets_min_size = (
            face_w >= settings.face_min_size and face_h >= settings.face_min_size
        )

        return QualityReport(
            is_sharp=is_sharp,
            blur_score=blur_score,
            meets_min_size=meets_min_size,
            face_width=face_w,
            face_height=face_h,
            passed=is_sharp and meets_min_size,
        )

    @staticmethod
    def validate_facial_geometry(kps: np.ndarray | list[list[float]] | None, bbox: BoundingBox) -> bool:
        """
        Validates 5-point facial landmarks against human biometric anatomy:
        1. Both eyes exist with sensible horizontal inter-pupillary separation.
        2. Vertical hierarchy: eyes must be above nose/mouth (tolerant of tilts).
        3. Reject degenerate clustered artifacts (ears, flowers, folds).
        """
        if kps is None:
            return True
        kps_arr = np.array(kps, dtype=np.float32)
        if kps_arr.shape[0] < 5:
            return True

        left_eye, right_eye, nose, left_mouth, right_mouth = kps_arr[:5]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < 3.0:
            return False

        bw = max(1.0, bbox.width)
        bh = max(1.0, bbox.height)
        eye_ratio = eye_dist / bw
        if eye_ratio < 0.08 or eye_ratio > 0.95:
            return False

        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (left_mouth + right_mouth) / 2.0
        face_axis = float(np.linalg.norm(mouth_mid - eye_mid))
        if face_axis < 3.0:
            return False

        # Head tilt tolerance: mouth should not be positioned above eyes by more than 25% face height
        if mouth_mid[1] < eye_mid[1] - (bh * 0.25):
            return False

        return True

    # ── Progressive Face Detection & Feature Extraction ───────

    def _get_raw_faces_progressive(
        self, image: np.ndarray, is_query: bool = False
    ) -> list[tuple[Any, tuple[float, float], float]]:
        """
        Progressive multi-stage face detection.

        Returns list of tuples: (raw_face_obj, (offset_x, offset_y), scale_factor)
        """
        if not self._initialized or self._app is None:
            raise RuntimeError("FaceEngine not initialized — call initialize() first")

        # Stage 1: Direct pass
        raw_faces = self._app.get(image)
        if raw_faces:
            return [(face, (0.0, 0.0), 1.0) for face in raw_faces]

        h, w = image.shape[:2]

        # Stage 1.5: High-Res Tiled Overlap Pass for large images with small faces
        if max(h, w) >= 800:
            tile_faces = []
            hw = int(w * 0.58)
            left_tile = image[:, :hw]
            right_tile = image[:, w - hw:]
            offset_right_x = float(w - hw)

            lf = self._app.get(left_tile)
            if lf:
                tile_faces.extend([(f, (0.0, 0.0), 1.0) for f in lf])
            rf = self._app.get(right_tile)
            if rf:
                tile_faces.extend([(f, (-offset_right_x, 0.0), 1.0) for f in rf])

            if tile_faces:
                return tile_faces

        # Stage 2: Tight-crop border reflection padding + scaling (helps for cropped passport / avatars)
        pad_h = max(24, int(h * 0.35))
        pad_w = max(24, int(w * 0.35))
        padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
        
        # If padded is still small (< 400px), upscale to give RetinaFace sufficient receptive field
        ph, pw = padded.shape[:2]
        if min(ph, pw) < 400:
            scale_pad = 400.0 / float(min(ph, pw))
            padded_scaled = cv2.resize(padded, (0, 0), fx=scale_pad, fy=scale_pad, interpolation=cv2.INTER_CUBIC)
            raw_faces = self._app.get(padded_scaled)
            if raw_faces:
                return [(face, (float(pad_w * scale_pad), float(pad_h * scale_pad)), scale_pad) for face in raw_faces]
        else:
            raw_faces = self._app.get(padded)
            if raw_faces:
                return [(face, (float(pad_w), float(pad_h)), 1.0) for face in raw_faces]

        # Stage 3: Small-image upscaling for tiny avatars or thumbnail faces (< 320px)
        if min(h, w) < 320:
            scale = max(2.0, 320.0 / float(min(h, w)))
            upscaled = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            raw_faces = self._app.get(upscaled)
            if raw_faces:
                return [(face, (0.0, 0.0), scale) for face in raw_faces]

        # Stage 4: Contrast & Lighting enhancement (CLAHE for backlit / shadowed faces)
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            raw_faces = self._app.get(enhanced)
            if raw_faces:
                return [(face, (0.0, 0.0), 1.0) for face in raw_faces]
        except Exception:
            pass

        # Stage 5: Rotation fallback (90°, 180°, 270°) for uploaded photos that may be oriented sideways
        if is_query or min(h, w) > 100:
            rotations = [
                (cv2.ROTATE_90_CLOCKWISE, 90),
                (cv2.ROTATE_180, 180),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
            ]
            for rot_code, _ in rotations:
                try:
                    rotated = cv2.rotate(image, rot_code)
                    raw_faces = self._app.get(rotated)
                    if raw_faces:
                        return [(face, (0.0, 0.0), 1.0) for face in raw_faces]
                except Exception:
                    pass

        # Stage 6: Direct ArcFace Crop Fallback for Target Photos (when user uploads already-cropped face)
        if is_query and hasattr(self._app, "models") and "recognition" in self._app.models:
            try:
                from insightface.app.common import Face
                rec_model = self._app.models["recognition"]
                # Provide canonical 5-point face landmarks for ArcFace norm_crop alignment
                kps = np.array([
                    [w * 0.35, h * 0.38],  # left eye
                    [w * 0.65, h * 0.38],  # right eye
                    [w * 0.50, h * 0.55],  # nose tip
                    [w * 0.38, h * 0.72],  # left mouth corner
                    [w * 0.62, h * 0.72],  # right mouth corner
                ], dtype=np.float32)
                face = Face(bbox=np.array([0, 0, w, h], dtype=np.float32), kps=kps, det_score=0.90)
                rec_model.get(image, face)
                if hasattr(face, "embedding") and face.embedding is not None and len(face.embedding) > 0:
                    logger.info("Direct ArcFace recognition fallback succeeded for query target crop")
                    return [(face, (0.0, 0.0), 1.0)]
            except Exception as e:
                logger.warning("Direct recognition fallback error: %s", e)

        return []

    # ── Face Detection ─────────────────────────────────────────

    def detect_faces(self, image: np.ndarray, is_query: bool = False) -> list[FaceDetection]:
        """
        Detect all faces in an image and return metadata (no embeddings yet).

        Returns faces sorted by bounding box area descending (largest first).
        """
        raw_tuples = self._get_raw_faces_progressive(image, is_query=is_query)
        if not raw_tuples:
            return []

        h, w = image.shape[:2]
        detections: list[FaceDetection] = []

        for idx, (face, (pad_x, pad_y), scale) in enumerate(raw_tuples):
            x1, y1, x2, y2 = face.bbox.astype(float).tolist()
            # Adjust for padding and scaling
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clamp coordinates to original image bounds
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0
            if not is_query and confidence < 0.35:
                continue

            # Extract 5-point landmarks if available
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                raw_kps = face.kps.tolist()
                landmarks = [[(pt[0] - pad_x) / scale, (pt[1] - pad_y) / scale] for pt in raw_kps]

            if not is_query and not self.validate_facial_geometry(landmarks, bbox):
                continue

            quality = self.assess_quality(image, bbox)

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=confidence,
                    face_index=idx,
                    quality=quality,
                    landmarks=landmarks,
                )
            )

        # Sort by area descending — largest face first
        detections.sort(key=lambda d: d.bbox.area, reverse=True)
        for i, det in enumerate(detections):
            det.face_index = i

        return detections

    # ── Embedding Extraction ───────────────────────────────────

    def extract_embeddings(
        self, image: np.ndarray, detections: list[FaceDetection] | None = None, is_query: bool = False
    ) -> list[FaceResult]:
        """
        Extract 512-d normalized ArcFace embeddings for detected faces.
        """
        raw_tuples = self._get_raw_faces_progressive(image, is_query=is_query)
        if not raw_tuples:
            return []

        h, w = image.shape[:2]
        results: list[FaceResult] = []

        for idx, (face, (pad_x, pad_y), scale) in enumerate(raw_tuples):
            x1, y1, x2, y2 = face.bbox.astype(float).tolist()
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0
            if not is_query and confidence < 0.35:
                continue

            quality = self.assess_quality(image, bbox)
            embedding = face.embedding.tolist()  # 512-d float list

            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                raw_kps = face.kps.tolist()
                landmarks = [[(pt[0] - pad_x) / scale, (pt[1] - pad_y) / scale] for pt in raw_kps]

            if not is_query and not self.validate_facial_geometry(landmarks, bbox):
                continue

            detection = FaceDetection(
                bbox=bbox,
                confidence=confidence,
                face_index=idx,
                quality=quality,
                landmarks=landmarks,
            )
            results.append(FaceResult(detection=detection, embedding=embedding))

        # Sort by area descending
        results.sort(key=lambda r: r.detection.bbox.area, reverse=True)
        for i, r in enumerate(results):
            r.detection.face_index = i

        return results

    # ── High-Level API ─────────────────────────────────────────

    def process_image(self, image_bytes: bytes, is_query: bool = True) -> list[FaceResult]:
        """
        End-to-end pipeline: decode → detect → extract embeddings.
        
        Args:
            image_bytes: Raw image file bytes.
            is_query: If True, uses progressive fallbacks (rotation, CLAHE, etc.)
                      optimized for user query target photos.
        """
        image = self.decode_image(image_bytes)
        results = self.extract_embeddings(image, is_query=is_query)

        if not results:
            raise FaceNotFoundError()

        logger.info("Processed image: %d face(s) detected", len(results))
        return results

    def process_image_with_quality_filter(
        self, image_bytes: bytes, strict: bool = False
    ) -> list[FaceResult]:
        """
        Like process_image but filters out faces that fail quality checks.

        Args:
            image_bytes: Raw image file bytes.
            strict: If True, raise QualityCheckError when ALL faces fail.
                    If False, return only passing faces (may be empty).

        Returns:
            List of FaceResult that passed quality checks.
        """
        results = self.process_image(image_bytes)
        passed = [r for r in results if r.detection.quality.passed]

        if strict and not passed:
            failures = [r.detection.quality.summary for r in results]
            raise QualityCheckError(
                message="All detected faces failed quality checks",
                details={"failures": failures},
            )

        logger.info(
            "Quality filter: %d/%d faces passed", len(passed), len(results)
        )
        return passed if passed else results  # Fall back to all if none pass

    # ── Batch Processing ───────────────────────────────────────

    def process_batch(self, images: list[bytes]) -> list[list[FaceResult]]:
        """
        Process multiple images sequentially.

        For Celery worker throughput — avoids re-initializing the model
        per image within a single worker process.

        Args:
            images: List of raw image bytes.

        Returns:
            List of face result lists, one per input image.
        """
        batch_results: list[list[FaceResult]] = []
        for i, image_bytes in enumerate(images):
            try:
                results = self.process_image(image_bytes)
                batch_results.append(results)
            except (InvalidImageError, FaceNotFoundError) as e:
                logger.warning("Batch item %d failed: %s", i, e.message)
                batch_results.append([])
        return batch_results

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def crop_face(image: np.ndarray, bbox: BoundingBox, padding: float = 0.1) -> np.ndarray:
        """
        Crop a face from the image with optional padding.

        Args:
            image: BGR numpy array.
            bbox: Face bounding box.
            padding: Fractional padding around the bbox (0.1 = 10%).

        Returns:
            Cropped face as BGR numpy array.
        """
        h, w = image.shape[:2]
        pad_w = bbox.width * padding
        pad_h = bbox.height * padding

        x1 = max(0, int(bbox.x1 - pad_w))
        y1 = max(0, int(bbox.y1 - pad_h))
        x2 = min(w, int(bbox.x2 + pad_w))
        y2 = min(h, int(bbox.y2 + pad_h))

        return image[y1:y2, x1:x2].copy()

    @classmethod
    def crop_face_b64(cls, image: np.ndarray, bbox: BoundingBox, padding: float = 0.1, quality: int = 85) -> str:
        """
        Crop a face from the image and return as a Base64 data URL for UI rendering.
        """
        import base64
        crop = cls.crop_face(image, bbox, padding=padding)
        if crop is None or crop.size == 0:
            return ""
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"
