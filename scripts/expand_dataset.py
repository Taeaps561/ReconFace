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
META_FILE = DATASET_DIR / "metadata.json"

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

async def expand_dataset(max_pages_per_category: int = 25, max_new_images: int = 5000):
    print("================================================================")
    print("🚀 SUAN DUSIT DATASET EXPANDER (CRAWL & DOWNLOAD BROADER ARCHIVES)")
    print("================================================================")

    # Load existing images and metadata to avoid duplicates
    existing_meta = []
    existing_urls = set()
    next_id = 1

    if META_FILE.exists():
        try:
            existing_meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            for m in existing_meta:
                if "image_url" in m:
                    existing_urls.add(normalize_key(m["image_url"]))
                next_id = max(next_id, m.get("id", 0) + 1)
            print(f"📦 Found existing dataset: {len(existing_meta)} images already on SSD.")
        except Exception as e:
            print(f"Warning reading existing metadata: {e}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    discovered_images = {}
    article_urls = set()

    # Category and archive listing pages to crawl
    listing_sources = []
    # General homepage pagination
    for p in range(1, max_pages_per_category + 1):
        listing_sources.append(f"https://www.dusit.ac.th/home/page/{p}")
    # Activity and news categories
    for p in range(1, max_pages_per_category + 1):
        listing_sources.append(f"https://www.dusit.ac.th/home/category/%e0%b8%82%e0%b9%88%e0%b8%b2%e0%b8%a7%e0%b8%81%e0%b8%b4%e0%b8%88%e0%b8%81%e0%b8%a3%e0%b8%a3%e0%b8%a1/page/{p}")
        listing_sources.append(f"https://www.dusit.ac.th/home/category/%e0%b8%82%e0%b9%88%e0%b8%b2%e0%b8%a7%e0%b8%9b%e0%b8%a3%e0%b8%b0%e0%b8%8a%e0%b8%b2%e0%b8%a3%e0%b8%b1%e0%b8%a1%e0%b8%9e%e0%b8%b1%e0%b8%99%e0%b8%98%e0%b9%8c/page/{p}")

    print(f"\n🔍 Phase 1: Scanning {len(listing_sources)} listing and archive pages for articles...")

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers, limits=httpx.Limits(max_connections=30, max_keepalive_connections=20)) as client:
        sem_list = asyncio.Semaphore(12)

        async def fetch_listing_page(url: str):
            async with sem_list:
                try:
                    res = await client.get(url)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, "html.parser")
                        for a in soup.find_all("a", href=True):
                            href = a["href"].strip()
                            abs_url = urljoin(url, href)
                            if "dusit.ac.th" in abs_url:
                                if re.search(r"/\d{4}/\d+", abs_url) or any(k in abs_url.lower() for k in ["/news/", "/activity/", "/post/"]):
                                    if not any(abs_url.lower().endswith(ext) for ext in [".pdf", ".jpg", ".png", ".zip", ".doc"]):
                                        article_urls.add(abs_url)
                except Exception:
                    pass

        await asyncio.gather(*[fetch_listing_page(u) for u in listing_sources])
        print(f"✅ Found {len(article_urls)} unique articles across Suan Dusit archives!")

        # Phase 2: Extract image galleries from discovered articles
        print(f"\n🖼️ Phase 2: Inspecting article galleries for full-resolution photos...")
        sem_art = asyncio.Semaphore(15)

        async def fetch_article_images(art_url: str):
            async with sem_art:
                try:
                    res = await client.get(art_url)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, "html.parser")
                        title = soup.title.string.strip() if soup.title and soup.title.string else art_url

                        for img in soup.find_all("img"):
                            for attr in ("src", "data-src", "data-original", "data-large-file", "data-full-image"):
                                val = img.get(attr)
                                if val and not val.startswith("data:"):
                                    u = unwrap_highres_url(urljoin(art_url, val))
                                    k = normalize_key(u)
                                    if k not in existing_urls and k not in discovered_images:
                                        discovered_images[k] = {"image_url": u, "page_url": art_url, "page_title": title}

                        for a in soup.find_all("a", href=True):
                            href = a["href"].strip()
                            abs_href = urljoin(art_url, href)
                            if any(abs_href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]) or "/gallery/" in abs_href.lower() or "/uploads/" in abs_href.lower():
                                u = unwrap_highres_url(abs_href)
                                k = normalize_key(u)
                                if k not in existing_urls and k not in discovered_images:
                                    discovered_images[k] = {"image_url": u, "page_url": art_url, "page_title": title}
                except Exception:
                    pass

        art_list = list(article_urls)
        for i in range(0, len(art_list), 30):
            chunk = art_list[i : i + 30]
            await asyncio.gather(*[fetch_article_images(u) for u in chunk])
            print(f" -> Inspected {min(i + 30, len(art_list))}/{len(art_list)} articles (New unique images found: {len(discovered_images)})...", flush=True)
            if len(discovered_images) >= max_new_images:
                print(f"Reached limit of {max_new_images} new images.")
                break

        print(f"\n🎉 Total New Unique Images to Download: {len(discovered_images)} images!")
        if not discovered_images:
            print("No new images to download. Dataset is up to date.")
            return

        # Phase 3: Fast concurrent download
        print(f"\n📥 Phase 3: Downloading new images concurrently into {IMAGES_DIR}...")
        sem_dl = asyncio.Semaphore(40)
        download_success = 0
        new_meta = []

        async def download_one(idx: int, item: dict):
            nonlocal download_success
            img_url = item["image_url"]
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            filename = f"img_{idx:04d}{ext}"
            file_path = IMAGES_DIR / filename

            if file_path.exists() and file_path.stat().st_size > 1000:
                download_success += 1
                return {
                    "id": idx,
                    "filename": filename,
                    "file_path": str(file_path.resolve()),
                    "image_url": img_url,
                    "page_url": item["page_url"],
                    "page_title": item["page_title"],
                    "size_bytes": file_path.stat().st_size,
                }

            async with sem_dl:
                try:
                    r = await client.get(img_url, timeout=10.0)
                    if r.status_code == 200 and len(r.content) > 1000:
                        file_path.write_bytes(r.content)
                        download_success += 1
                        if download_success % 50 == 0:
                            print(f" -> Downloaded {download_success}/{len(discovered_images)} new images...", flush=True)
                        return {
                            "id": idx,
                            "filename": filename,
                            "file_path": str(file_path.resolve()),
                            "image_url": img_url,
                            "page_url": item["page_url"],
                            "page_title": item["page_title"],
                            "size_bytes": len(r.content),
                        }
                except Exception:
                    pass
            return None

        items = list(discovered_images.values())[:max_new_images]
        tasks = [download_one(next_id + i, it) for i, it in enumerate(items)]
        downloaded = await asyncio.gather(*tasks)
        valid_downloaded = [d for d in downloaded if d is not None]

        # Combine existing and new metadata
        combined_meta = existing_meta + valid_downloaded
        META_FILE.write_text(json.dumps(combined_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n================================================================")
        print("🎉 DATASET EXPANSION COMPLETE!")
        print(f"📁 Previous images: {len(existing_meta)}")
        print(f"➕ Newly added images: {len(valid_downloaded)}")
        print(f"🏆 Total images now on SSD: {len(combined_meta)} images")
        print("================================================================")

if __name__ == "__main__":
    pages = 20
    if len(sys.argv) > 1:
        try:
            pages = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(expand_dataset(max_pages_per_category=pages))
