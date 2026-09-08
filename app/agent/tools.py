"""
Agent Tool Definitions for LangChain / CrewAI / OpenAI Function Calling.

Provides standardized, ready-to-use tools that allow autonomous AI Agents
to interact with the ReconFace Reverse Face Search System and perform OSINT tasks.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import httpx
from pydantic import BaseModel, Field


class FaceSearchResult(BaseModel):
    point_id: str
    score: float
    source_url: str
    thumbnail_path: str
    indexed_at: str


class ReconFaceTools:
    """
    Toolkit for AI Agents to interact with ReconFace API.
    Can be used standalone, or wrapped into LangChain/CrewAI tools.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")

    def search_face(
        self,
        image_input: str | bytes | Path,
        top_k: int = 5,
        score_threshold: float = 0.6,
        face_index: int = 0,
    ) -> dict[str, Any]:
        """
        Reverse face search tool.

        Args:
            image_input: File path (str/Path) or raw image bytes.
            top_k: Maximum number of visual matches to retrieve.
            score_threshold: Cosine similarity cutoff (0.0 to 1.0).
            face_index: Index of target face in image (0 = largest face).

        Returns:
            Dict containing detected faces summary and top matched URLs/metadata.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {"error": f"Image file not found: {path}"}
            image_bytes = path.read_bytes()
            filename = path.name
        else:
            image_bytes = image_input
            filename = "query_face.jpg"

        url = f"{self.api_base_url}/api/v1/search"
        params = {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "face_index": face_index,
        }
        files = {"file": (filename, io.BytesIO(image_bytes), "image/jpeg")}

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, params=params, files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": f"Face search failed: {str(e)}"}

    def ingest_url(
        self,
        image_url: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Enqueue an image URL for background face extraction and vector indexing.

        Args:
            image_url: URL of the target image.
            metadata: Custom tags/case ID to attach to vector payload.

        Returns:
            Dict with Celery task ID and queuing status.
        """
        url = f"{self.api_base_url}/api/v1/ingest"
        payload = {
            "url": image_url,
            "metadata": metadata or {},
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": f"Ingestion trigger failed: {str(e)}"}

    def scan_webpage_for_target(
        self,
        image_input: str | bytes | Path,
        webpage_url: str,
        score_threshold: float = 0.50,
    ) -> dict[str, Any]:
        """
        Actively crawls a target website, detects faces across all pages & galleries,
        and matches against the target face.

        Args:
            image_input: Target face image file path or raw bytes.
            webpage_url: Target website or article URL to crawl and scan.
            score_threshold: Similarity cutoff.

        Returns:
            Dict containing matched images, scores, and source sub-page URLs.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {"error": f"Image file not found: {path}"}
            image_bytes = path.read_bytes()
            filename = path.name
        else:
            image_bytes = image_input
            filename = "target_face.jpg"

        url = f"{self.api_base_url}/api/v1/scan-webpage-for-target"
        params = {
            "webpage_url": webpage_url,
            "score_threshold": score_threshold,
        }
        files = {"file": (filename, io.BytesIO(image_bytes), "image/jpeg")}

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, params=params, files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": f"Webpage scan failed: {str(e)}"}


# ── Optional LangChain Wrapper ─────────────────────────────────
try:
    from langchain.tools import tool

    recon_tools = ReconFaceTools()

    @tool
    def search_face_tool(image_path: str, score_threshold: float = 0.65) -> str:
        """
        Reverse searches a face image against indexed OSINT visual databases.
        Input is the local file path to the face image.
        Returns matched URLs, similarity scores, and metadata.
        """
        res = recon_tools.search_face(image_path, score_threshold=score_threshold)
        return str(res)

    @tool
    def ingest_face_url_tool(image_url: str, case_tag: str = "osint-investigation") -> str:
        """
        Asynchronously downloads and indexes all faces found in a target image URL.
        Use this when you discover new photo URLs on social media or forums.
        """
        res = recon_tools.ingest_url(image_url, metadata={"case_tag": case_tag})
        return str(res)

except ImportError:
    pass
