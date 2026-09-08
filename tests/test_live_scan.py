import asyncio
import json
from pathlib import Path
import httpx

async def run_live_test():
    root_dir = Path(__file__).resolve().parent.parent
    target_path = root_dir / "test_target.jpg"
    print("Testing live SSE /api/v1/scan-stream with target:", target_path.name)
    
    url = "http://127.0.0.1:8000/api/v1/scan-stream?webpage_url=https%3A%2F%2Fwww.dusit.ac.th%2Fhome%2F%3Fs%3D%E0%B8%A7%E0%B8%B4%E0%B8%8A%E0%B8%8A%E0%B8%B2%2B%E0%B8%89%E0%B8%B4%E0%B8%A1%E0%B8%9E%E0%B8%A5%E0%B8%B5&score_threshold=0.55&max_pages=5"
    
    files = {"file": ("target.jpg", target_path.read_bytes(), "image/jpeg")}
    matches_received = []
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", url, files=files) as response:
            print("HTTP Response Status:", response.status_code)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        payload = json.loads(data_str)
                        ev_type = payload.get("type")
                        if ev_type == "status":
                            print(" -> Status:", payload.get("message"), flush=True)
                        elif ev_type == "crawl_done":
                            print(f" -> Crawl Done: {payload.get('pages_crawled')} pages, {payload.get('total_images')} images", flush=True)
                        elif ev_type == "scan_progress":
                            print(f" -> Progress: {payload.get('scanned')}/{payload.get('total')} (Matches: {payload.get('matches_count')})", flush=True)
                        elif ev_type == "match_found":
                            m = payload.get("match", {})
                            score = m.get("score", 0)
                            title = m.get("page_title", "")
                            img = m.get("image_url", "")
                            has_crop = bool(m.get("thumbnail_b64"))
                            matches_received.append((score, title, img))
                            print(f" 🎯 MATCH #{payload.get('count')}: {score*100:.1f}% on '{title[:35]}' (Has Crop: {has_crop})", flush=True)
                        elif ev_type == "complete":
                            print(f" -> Complete! Matched count: {payload.get('matched_count')}, Duration: {payload.get('duration')}s", flush=True)
                    except Exception:
                        pass

    print("\n================ LIVE TEST SUMMARY ================")
    print(f"Total verified matches: {len(matches_received)}")
    for idx, (sc, tt, im) in enumerate(matches_received, 1):
        print(f"  {idx}. {sc*100:.1f}% Match | {tt} | {im[-35:]}")

if __name__ == "__main__":
    asyncio.run(run_live_test())
