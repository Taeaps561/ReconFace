import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset_dusit"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = DATASET_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

WP_DIM_REGEX = re.compile(r"-(?:\d+x\d+|scaled|rotated|large|medium|small|thumb)(?=\.[a-zA-Z0-9]+(?:\?.*)?$)", re.IGNORECASE)

def unwrap_highres_url(url: str) -> str:
    if "/thumbs/thumbs_" in url:
        url = url.replace("/thumbs/thumbs_", "/")
    elif "/thumbs/" in url:
        url = url.replace("/thumbs/", "/")
    return WP_DIM_REGEX.sub("", url)

def normalize_key(url: str) -> str:
    try:
        p = urlparse(url)
        clean = WP_DIM_REGEX.sub("", p.path).replace("/thumbs/thumbs_", "/").replace("/thumbs/", "/")
        return f"{p.scheme}://{p.netloc.lower()}{clean}"
    except Exception:
        return url

async def main():
    print("🚀 FAST DATASET SCRAPER FOR SUAN DUSIT (Target: อ.วิชชา ฉิมพลี)")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    discovered_images = {}  # norm_key -> {image_url, page_url, page_title}
    article_urls = set()

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers, limits=httpx.Limits(max_connections=30, max_keepalive_connections=20)) as client:
        # Phase 1: Fetch all 14 search result pagination pages
        print("\n🔍 Phase 1: Fetching all 14 search result pages...")
        search_pages = [f"https://www.dusit.ac.th/home/page/{p}?s=วิชชา+ฉิมพลี" for p in range(1, 15)]
        
        async def fetch_search_page(url: str):
            try:
                res = await client.get(url)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, "html.parser")
                    # Extract articles
                    for a in soup.find_all("a", href=True):
                        href = a["href"].strip()
                        abs_url = urljoin(url, href)
                        if "dusit.ac.th" in abs_url:
                            # Match post / article URLs
                            if re.search(r"/\d{4}/\d+", abs_url) or any(k in abs_url.lower() for k in ["/news/", "/activity/", "/post/"]):
                                if not any(abs_url.lower().endswith(ext) for ext in [".pdf", ".jpg", ".png", ".zip", ".doc"]):
                                    article_urls.add(abs_url)
                    
                    # Extract thumbnail images directly from search results
                    for img in soup.find_all("img"):
                        for attr in ("src", "data-src", "data-original"):
                            val = img.get(attr)
                            if val and not val.startswith("data:"):
                                u = unwrap_highres_url(urljoin(url, val))
                                k = normalize_key(u)
                                if k not in discovered_images:
                                    discovered_images[k] = {"image_url": u, "page_url": url, "page_title": "Search Results"}
            except Exception as e:
                print(f"Error fetching search page {url}: {e}")

        await asyncio.gather(*[fetch_search_page(u) for u in search_pages])
        print(f"✅ Found {len(article_urls)} relevant articles across 14 search pages!")

        # Phase 2: Concurrently crawl all article pages to extract full galleries
        print(f"\n🖼️ Phase 2: Extracting high-res image galleries from {len(article_urls)} articles...")
        
        async def fetch_article_images(art_url: str):
            try:
                res = await client.get(art_url)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else art_url

                    # Extract all <img> and lazy images
                    for img in soup.find_all("img"):
                        for attr in ("src", "data-src", "data-original", "data-large-file", "data-full-image"):
                            val = img.get(attr)
                            if val and not val.startswith("data:"):
                                u = unwrap_highres_url(urljoin(art_url, val))
                                k = normalize_key(u)
                                if k not in discovered_images:
                                    discovered_images[k] = {"image_url": u, "page_url": art_url, "page_title": title}

                    # Extract NextGEN and large image href links
                    for a in soup.find_all("a", href=True):
                        href = a["href"].strip()
                        abs_href = urljoin(art_url, href)
                        if any(abs_href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]) or "/gallery/" in abs_href.lower() or "/uploads/" in abs_href.lower():
                            u = unwrap_highres_url(abs_href)
                            k = normalize_key(u)
                            if k not in discovered_images:
                                discovered_images[k] = {"image_url": u, "page_url": art_url, "page_title": title}
            except Exception:
                pass

        # Batch articles in chunks of 15
        art_list = list(article_urls)
        for i in range(0, len(art_list), 15):
            chunk = art_list[i : i + 15]
            await asyncio.gather(*[fetch_article_images(u) for u in chunk])
            print(f" -> Processed {min(i + 15, len(art_list))}/{len(art_list)} articles (Discovered: {len(discovered_images)} images)...", flush=True)

        print(f"\n🎉 Total Unique Images Discovered: {len(discovered_images)} images!")
        print(f"📥 Phase 3: Downloading all images concurrently to {IMAGES_DIR}...")

        # Phase 3: Fast concurrent image downloading
        download_count = 0
        metadata = []
        sem = asyncio.Semaphore(60)

        async def download_image(idx: int, item: dict):
            nonlocal download_count
            img_url = item["image_url"]
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            filename = f"img_{idx:04d}{ext}"
            file_path = IMAGES_DIR / filename

            # If already downloaded, use cached file
            if file_path.exists() and file_path.stat().st_size > 1000:
                download_count += 1
                return {
                    "id": idx,
                    "filename": filename,
                    "file_path": str(file_path.resolve()),
                    "image_url": img_url,
                    "page_url": item["page_url"],
                    "page_title": item["page_title"],
                    "size_bytes": file_path.stat().st_size,
                }

            async with sem:
                try:
                    res = await client.get(img_url, timeout=8.0)
                    if res.status_code == 200 and len(res.content) > 1000:
                        file_path.write_bytes(res.content)
                        download_count += 1
                        if download_count % 100 == 0 or download_count == len(discovered_images):
                            print(f" -> Saved {download_count}/{len(discovered_images)} images to disk...", flush=True)

                        return {
                            "id": idx,
                            "filename": filename,
                            "file_path": str(file_path.resolve()),
                            "image_url": img_url,
                            "page_url": item["page_url"],
                            "page_title": item["page_title"],
                            "size_bytes": len(res.content),
                        }
                except Exception:
                    pass
            return None

        items = list(discovered_images.values())
        tasks = [download_image(i + 1, it) for i, it in enumerate(items)]
        results = await asyncio.gather(*tasks)
        metadata = [r for r in results if r is not None]

        meta_path = DATASET_DIR / "metadata.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n===========================================================")
        print(f"🎉 OFFLINE DATASET CREATION COMPLETE!")
        print(f"📁 Images saved: {len(metadata)} images in {IMAGES_DIR}")
        print(f"📄 Metadata saved: {meta_path}")
        print("===========================================================")

if __name__ == "__main__":
    asyncio.run(main())
