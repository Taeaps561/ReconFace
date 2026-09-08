"""
Recursive Full-Website Crawler and Image Extractor for OSINT Face Reconnaissance.
Crawls entire domains with BFS queue, extracts all images across internal links,
rich media layouts, lazy-load attributes, CSS styles, and JSON-LD structured data.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import re
from urllib.parse import parse_qs, unquote, urldefrag, urljoin, urlparse
from typing import Any
import httpx
from bs4 import BeautifulSoup

from app.core.logging import get_logger

logger = get_logger("web_crawler")


class FullSiteCrawlerService:
    """Recursive BFS Crawler that scans multiple pages across an entire website."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "th,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }

    IGNORE_EXTENSIONS = {
        ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".avi", ".mov",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".exe", ".apk",
        ".css", ".js", ".json", ".xml", ".txt"
    }

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".tiff", ".tif", ".avif"
    }

    CSS_BG_REGEX = re.compile(
        r"""background(?:-image)?\s*:\s*[^;]*?url\(\s*['"]?(.*?)['"]?\s*\)""",
        re.IGNORECASE
    )

    @classmethod
    async def extract_images_from_page(
        cls,
        page_url: str,
        max_images: int = 50,
    ) -> tuple[str, list[str]]:
        """
        Extract all image URLs from a single web page.
        
        Returns:
            Tuple of (page_title, list_of_image_urls)
        """
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers=cls.HEADERS,
        ) as client:
            try:
                resp = await client.get(page_url)
                if resp.status_code != 200:
                    return page_url, []
                soup = BeautifulSoup(resp.text, "html.parser")
                page_title = (
                    soup.title.string.strip()
                    if soup.title and soup.title.string
                    else page_url
                )
                images = list(cls._extract_images_from_soup(soup, page_url))
                return page_title, images[:max_images]
            except Exception as e:
                logger.warning("Failed to extract images from page %s: %s", page_url, e)
                return page_url, []

    @classmethod
    async def crawl_website(
        cls,
        start_url: str,
        max_pages: int = 150,
        max_depth: int = 3,
        max_images: int | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Crawls a website starting from start_url using BFS and collects all relevant images found.
        
        Args:
            start_url: Target URL to begin crawl from.
            max_pages: Maximum number of distinct pages to crawl (default 150).
            max_depth: Maximum link depth to follow.
            max_images: Optional cap on unique images collected (None for unlimited).

        Returns:
            Tuple of (main_title, list of dicts with {"image_url": ..., "page_url": ..., "page_title": ...})
        """
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc.lower()

        visited_urls: set[str] = set()
        discovered_images: list[dict[str, str]] = []
        seen_img_urls: set[str] = set()

        # BFS queue: (url, current_depth)
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        visited_urls.add(cls._normalize_url(start_url))

        main_title = start_url
        crawled_count = 0
        sem = asyncio.Semaphore(12)

        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers=cls.HEADERS,
            limits=httpx.Limits(max_connections=35, max_keepalive_connections=25),
        ) as client:
            async def fetch_page(url: str, depth: int):
                async with sem:
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            return url, depth, None, None
                        content_type = resp.headers.get("content-type", "")
                        if "text/html" not in content_type and "application/xhtml" not in content_type:
                            return url, depth, None, None
                        soup = BeautifulSoup(resp.text, "html.parser")
                        title = soup.title.string.strip() if soup.title and soup.title.string else url
                        return url, depth, soup, title
                    except Exception as ex:
                        logger.debug("Error crawling %s: %s", url, ex)
                        return url, depth, None, None

            is_start_search = any(k in start_url.lower() for k in ["?s=", "?q=", "search"])

            while queue and (max_pages is None or crawled_count < max_pages):
                if max_images and len(discovered_images) >= max_images:
                    break

                # Pop a batch of URLs to process concurrently
                batch = []
                while queue and len(batch) < 10 and (max_pages is None or (crawled_count + len(batch)) < max_pages):
                    batch.append(queue.popleft())

                if not batch:
                    break

                crawled_count += len(batch)
                results = await asyncio.gather(*[fetch_page(u, d) for u, d in batch])

                for current_url, depth, soup, page_title in results:
                    if not soup:
                        continue

                    if current_url == start_url or main_title == start_url:
                        main_title = page_title

                    # 1. Extract ALL images on current page with article context and unwrapped high-res resolution
                    page_items = cls._extract_images_with_context(soup, current_url, page_title)
                    for it in page_items:
                        raw_img_url = it["raw_url"]
                        highres_url = cls._unwrap_highres_url(raw_img_url)
                        norm_key = cls._normalize_image_key(highres_url)
                        if norm_key not in seen_img_urls:
                            seen_img_urls.add(norm_key)
                            discovered_images.append({
                                "image_url": highres_url,
                                "raw_image_url": raw_img_url,
                                "page_url": it.get("article_url") or current_url,
                                "page_title": it.get("article_title") or page_title,
                            })
                            if max_images and len(discovered_images) >= max_images:
                                break

                    if max_images and len(discovered_images) >= max_images:
                        break

                    # 2. Extract internal links & pagination links
                    if depth < max_depth:
                        is_current_article = bool(
                            re.search(r"/\d{4}/\d+", current_url)
                            or any(k in current_url.lower() for k in ["/news/", "/article/", "/post/", "/activity/", "/gallery/"])
                        )
                        # If starting from a search query and on a single article, extract its images without drifting into irrelevant menus
                        if not (is_start_search and is_current_article and current_url != start_url):
                            for a in soup.find_all("a", href=True):
                                href = a["href"].strip()
                                abs_url = urljoin(current_url, href)
                                clean_url = cls._normalize_url(abs_url)

                                if (
                                    cls._is_valid_internal_link(clean_url, base_domain)
                                    and clean_url not in visited_urls
                                ):
                                    visited_urls.add(clean_url)
                                    is_search_page = any(p in clean_url.lower() for p in ["?s=", "?q=", "search", "page/", "paged="])
                                    is_article_post = bool(
                                        re.search(r"/\d{4}/\d+", clean_url)
                                        or any(k in clean_url.lower() for k in ["/news", "/article", "/post", "/activity", "/view", "/gallery", "/pr-"])
                                    )

                                    if is_search_page:
                                        queue.appendleft((clean_url, depth))
                                    elif is_article_post:
                                        queue.append((clean_url, depth + 1))
                                    elif not is_start_search:
                                        queue.append((clean_url, depth + 1))

        logger.info(
            "Crawl finished: %d pages crawled (%d unique urls discovered), %d images collected from %s",
            crawled_count, len(visited_urls), len(discovered_images), start_url
        )
        return main_title, discovered_images

    WP_DIM_REGEX = re.compile(r"-(?:\d+x\d+|scaled|rotated|large|medium|small|thumb)(?=\.[a-zA-Z0-9]+(?:\?.*)?$)", re.IGNORECASE)

    @classmethod
    def _unwrap_highres_url(cls, url: str) -> str:
        """
        Unwraps thumbnail/responsive URLs to their high-resolution full image source:
        1. NextGEN gallery thumbs: .../thumbs/thumbs_DSC04595.jpg -> .../DSC04595.jpg
        2. WordPress responsive dimensions: photo-300x200.jpg -> photo.jpg
        """
        try:
            # 1. NextGEN gallery thumbnail unwrap
            if "/thumbs/thumbs_" in url:
                url = url.replace("/thumbs/thumbs_", "/")
            elif "/thumbs/" in url:
                url = url.replace("/thumbs/", "/")

            # 2. WordPress responsive dimensions unwrap
            clean_url = cls.WP_DIM_REGEX.sub("", url)
            return clean_url
        except Exception:
            return url

    @classmethod
    def _normalize_image_key(cls, url: str) -> str:
        """
        Normalize an image URL to its canonical base key by stripping WordPress/CMS
        dimension suffixes (-150x150, -300x200, -1024x768, -scaled) and resizing query params.
        """
        try:
            parsed = urlparse(url)
            # Remove dimensions from path: image-1024x768.jpg -> image.jpg
            clean_path = cls.WP_DIM_REGEX.sub("", parsed.path)
            clean_path = clean_path.replace("/thumbs/thumbs_", "/").replace("/thumbs/", "/")
            # Build canonical base key (ignore query string resizing parameters)
            canonical = f"{parsed.scheme}://{parsed.netloc.lower()}{clean_path}"
            return canonical
        except Exception:
            return url

    @classmethod
    def _select_best_from_srcset(cls, base_url: str, srcset_val: str) -> str | None:
        """Parse srcset and return the highest resolution candidate URL."""
        if not srcset_val:
            return None
        best_url = None
        best_size = -1

        for part in srcset_val.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            cand_url = tokens[0].strip()
            size = 0
            if len(tokens) > 1:
                desc = tokens[1].lower().strip()
                if desc.endswith("w"):
                    try:
                        size = int(desc[:-1])
                    except ValueError:
                        size = 0
                elif desc.endswith("x"):
                    try:
                        size = int(float(desc[:-1]) * 1000)
                    except ValueError:
                        size = 0
            abs_url = cls._resolve_url(base_url, cand_url)
            if cls._is_valid_image_url(abs_url):
                if size >= best_size:
                    best_size = size
                    best_url = abs_url

        return best_url

    @classmethod
    def _extract_images_with_context(cls, soup: BeautifulSoup, base_url: str, default_title: str) -> list[dict[str, str]]:
        """
        Extract candidate images along with their enclosing article page URL and headline title.
        """
        results: list[dict[str, str]] = []
        seen: set[str] = set()

        lazy_attributes = (
            "src", "data-src", "data-original", "data-lazy-src", "data-full-image",
            "data-orig-file", "data-large-file", "data-zoom-src", "data-high-res-src",
            "data-src-retina", "data-fallback-src", "data-url", "srcset", "data-srcset"
        )

        for img in soup.find_all("img"):
            # Check if img is wrapped in an <a> tag linking to an article
            parent_a = img.find_parent("a", href=True)
            parent_article = img.find_parent(["article", "div"], class_=re.compile(r"post|entry|article|news|event|card|item", re.I))
            
            article_url = base_url
            article_title = default_title

            if parent_a:
                href = parent_a["href"].strip()
                if href and not any(href.lower().endswith(ext) for ext in cls.IMAGE_EXTENSIONS) and not href.startswith("#") and not href.startswith("javascript:"):
                    resolved_a = cls._resolve_url(base_url, href)
                    if cls._is_valid_internal_link(resolved_a, urlparse(base_url).netloc.lower()):
                        article_url = resolved_a
                
                a_title = parent_a.get("title") or parent_a.get_text(strip=True)
                if a_title and len(a_title) > 4:
                    article_title = a_title

            if parent_article and (article_url == base_url or article_title == default_title):
                art_a = parent_article.find("a", href=True)
                if art_a:
                    href = art_a["href"].strip()
                    if href and not any(href.lower().endswith(ext) for ext in cls.IMAGE_EXTENSIONS):
                        resolved_a = cls._resolve_url(base_url, href)
                        if cls._is_valid_internal_link(resolved_a, urlparse(base_url).netloc.lower()):
                            article_url = resolved_a
                h_elem = parent_article.find(["h1", "h2", "h3", "h4", "h5", "a"])
                if h_elem:
                    h_txt = h_elem.get_text(strip=True)
                    if h_txt and len(h_txt) > 4:
                        article_title = h_txt

            img_title = img.get("alt") or img.get("title")
            if img_title and len(img_title) > 5 and article_title == default_title:
                article_title = img_title

            for attr in lazy_attributes:
                val = img.get(attr)
                if not val or val.startswith("data:"):
                    continue
                if "srcset" in attr:
                    for part in val.split(","):
                        candidate = part.strip().split()[0]
                        abs_url = cls._resolve_url(base_url, candidate)
                        if cls._is_valid_image_url(abs_url) and abs_url not in seen:
                            seen.add(abs_url)
                            results.append({
                                "raw_url": abs_url,
                                "article_url": article_url,
                                "article_title": article_title
                            })
                else:
                    abs_url = cls._resolve_url(base_url, val.strip())
                    if cls._is_valid_image_url(abs_url) and abs_url not in seen:
                        seen.add(abs_url)
                        results.append({
                            "raw_url": abs_url,
                            "article_url": article_url,
                            "article_title": article_title
                        })

        # Also grab any standalone backgrounds or meta image tags
        other_imgs = cls._extract_images_from_soup(soup, base_url)
        for raw_u in other_imgs:
            if raw_u not in seen:
                seen.add(raw_u)
                results.append({
                    "raw_url": raw_u,
                    "article_url": base_url,
                    "article_title": default_title
                })

        return results

    @classmethod
    def _extract_images_from_soup(cls, soup: BeautifulSoup, base_url: str) -> set[str]:
        """Extract all candidate image URLs from HTML markup."""
        image_set: set[str] = set()

        # 1. <img> tags (standard src, lazy-load attributes, and srcset)
        lazy_attributes = (
            "src", "data-src", "data-original", "data-lazy-src", "data-full-image",
            "data-orig-file", "data-large-file", "data-zoom-src", "data-high-res-src",
            "data-src-retina", "data-fallback-src", "data-url", "srcset", "data-srcset"
        )
        for img in soup.find_all("img"):
            for attr in lazy_attributes:
                val = img.get(attr)
                if not val or val.startswith("data:"):
                    continue
                if "srcset" in attr:
                    for part in val.split(","):
                        candidate = part.strip().split()[0]
                        abs_url = cls._resolve_url(base_url, candidate)
                        if cls._is_valid_image_url(abs_url):
                            image_set.add(abs_url)
                else:
                    abs_url = cls._resolve_url(base_url, val.strip())
                    if cls._is_valid_image_url(abs_url):
                        image_set.add(abs_url)

        # 2. <picture> and <source> elements
        for source in soup.find_all("source"):
            for attr in ("srcset", "src", "data-srcset", "data-src"):
                val = source.get(attr)
                if not val or val.startswith("data:"):
                    continue
                if "srcset" in attr:
                    for part in val.split(","):
                        candidate = part.strip().split()[0]
                        abs_url = cls._resolve_url(base_url, candidate)
                        if cls._is_valid_image_url(abs_url):
                            image_set.add(abs_url)
                else:
                    abs_url = cls._resolve_url(base_url, val.strip())
                    if cls._is_valid_image_url(abs_url):
                        image_set.add(abs_url)

        # 3. Inline style background images: <div style="background-image: url(...)">
        for elem in soup.find_all(style=True):
            style_content = elem.get("style", "")
            matches = cls.CSS_BG_REGEX.findall(style_content)
            for m in matches:
                clean_m = m.strip("'\" \t\r\n")
                if clean_m and not clean_m.startswith("data:"):
                    abs_url = cls._resolve_url(base_url, clean_m)
                    if cls._is_valid_image_url(abs_url):
                        image_set.add(abs_url)

        # 4. <a> tags linking directly to high-res image files
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("data:") or href.startswith("#") or href.startswith("javascript:"):
                continue
            
            abs_url = cls._resolve_url(base_url, href)
            path = urlparse(abs_url).path.lower()
            if any(path.endswith(ext) for ext in cls.IMAGE_EXTENSIONS):
                if cls._is_valid_image_url(abs_url):
                    image_set.add(abs_url)

        # 5. OpenGraph, Twitter, and Link Meta image tags
        for meta in soup.find_all("meta"):
            prop = (meta.get("property") or meta.get("name") or "").lower()
            if prop in (
                "og:image", "og:image:url", "og:image:secure_url",
                "twitter:image", "twitter:image:src"
            ):
                content = meta.get("content")
                if content and not content.startswith("data:"):
                    abs_url = cls._resolve_url(base_url, content.strip())
                    if cls._is_valid_image_url(abs_url):
                        image_set.add(abs_url)

        for link in soup.find_all("link"):
            rel = link.get("rel", "")
            if isinstance(rel, list):
                rel = " ".join(rel)
            if "image_src" in rel.lower() or "apple-touch-icon" in rel.lower():
                href = link.get("href")
                if href and not href.startswith("data:"):
                    abs_url = cls._resolve_url(base_url, href.strip())
                    if cls._is_valid_image_url(abs_url):
                        image_set.add(abs_url)

        # 6. JSON-LD structured data (<script type="application/ld+json">)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                if not script.string:
                    continue
                data = json.loads(script.string)
                cls._extract_images_from_json(data, base_url, image_set)
            except Exception:
                pass

        return image_set

    @classmethod
    def _extract_images_from_json(cls, data: Any, base_url: str, image_set: set[str]) -> None:
        """Recursively search for image URLs in JSON-LD objects."""
        if isinstance(data, dict):
            for k, v in data.items():
                if k.lower() in ("image", "thumbnailurl", "contenturl", "primaryimageofpage", "avatar", "photo"):
                    if isinstance(v, str):
                        abs_url = cls._resolve_url(base_url, v)
                        if cls._is_valid_image_url(abs_url):
                            image_set.add(abs_url)
                    elif isinstance(v, dict):
                        cls._extract_images_from_json(v, base_url, image_set)
                    elif isinstance(v, list):
                        cls._extract_images_from_json(v, base_url, image_set)
                else:
                    cls._extract_images_from_json(v, base_url, image_set)
        elif isinstance(data, list):
            for item in data:
                cls._extract_images_from_json(item, base_url, image_set)

    @classmethod
    def _resolve_url(cls, base_url: str, relative_url: str) -> str:
        """Handle protocol-relative URLs (//example.com/...) and standard relative URLs."""
        clean = relative_url.strip()
        if clean.startswith("//"):
            scheme = urlparse(base_url).scheme or "https"
            return f"{scheme}:{clean}"
        return urljoin(base_url, clean)

    @staticmethod
    def _normalize_url(url: str) -> str:
        clean, _ = urldefrag(url)
        return clean.rstrip("/")

    @classmethod
    def _is_valid_internal_link(cls, url: str, base_domain: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            
            link_domain = parsed.netloc.lower().split(":")[0]
            base_clean = base_domain.lower().split(":")[0]

            if not cls._is_same_organization(link_domain, base_clean):
                return False

            lowered = url.lower()
            if any(lowered.endswith(ext) for ext in cls.IGNORE_EXTENSIONS):
                return False
            if any(ign in lowered for ign in ["logout", "login", "signin", "signup", "wp-admin", "cart", "checkout"]):
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _is_same_organization(domain1: str, domain2: str) -> bool:
        if domain1 == domain2:
            return True
        p1 = domain1.split(".")
        p2 = domain2.split(".")
        root1 = ".".join(p1[-3:]) if len(p1) >= 3 and p1[-2] in ("ac", "co", "in", "or", "go") else ".".join(p1[-2:])
        root2 = ".".join(p2[-3:]) if len(p2) >= 3 and p2[-2] in ("ac", "co", "in", "or", "go") else ".".join(p2[-2:])
        return root1 == root2 or domain1.endswith("." + root2) or domain2.endswith("." + root1)

    @classmethod
    def _is_valid_image_url(cls, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in cls.IGNORE_EXTENSIONS):
                return False

            lowered = url.lower()
            # Filter out tiny icon dimension suffixes (e.g. -36x11.png, -24x20.jpg, -34x48.png)
            if re.search(r"-[1-5]\d{0,1}x[1-5]\d{0,1}\.", lowered):
                return False

            # Filter out SVG vector icons, 1x1 tracking pixels, spacers, emojis, small badges, and UI icons
            noise_patterns = (
                ".svg", ".ico", "pixel", "tracking", "spacer", "1x1", "blank.gif", "empty.gif",
                "/icon/", "/icons/", "/flags/", "/emoji/", "/emojis/", "wp-includes/images/",
                "favicon", "badge", "btn_", "button", "logo-small", "logo_small", "logo-mini",
                "arrow-", "chevron", "loading.gif", "spinner.gif", "facebook.png", "twitter.png",
                "line.png", "youtube.png", "instagram.png", "social/"
            )
            if any(pat in lowered for pat in noise_patterns):
                return False
            
            # Extract Next.js embedded image URLs if present
            if "/_next/image" in lowered and "url=" in lowered:
                qs = parse_qs(parsed.query)
                if "url" in qs:
                    target = unquote(qs["url"][0])
                    if target.startswith("http"):
                        return True

            return True
        except Exception:
            return False


# Backward compatibility alias
WebCrawlerService = FullSiteCrawlerService
