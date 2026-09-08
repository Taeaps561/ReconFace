# 2. Unwrap High-Resolution URLs in Crawler Pipeline

**Date**: 2026-08-27  
**Status**: Accepted

## Context & Decision
The crawler previously downloaded whatever `src` attribute was found in `<img>` tags on HTML web pages. In typical CMS sites (such as WordPress), thumbnails are generated with suffix patterns like `-150x150.jpg` or `-300x184.png`. In these thumbnail images, human faces are reduced to 10–25 pixels in height, which falls below the minimum receptive field of the RetinaFace detection backbone. Furthermore, anchor links (`<a>`) pointing to PDF documents were erroneously ingested as image candidates.

We decided to:
1. Strip CMS resolution suffixes (e.g. `-[0-9]+x[0-9]+` and `-scaled`) to unwrap URLs back to the original full-resolution media asset.
2. Maintain `raw_image_url` as an automatic fallback if the unwrapped URL returns an HTTP 404.
3. Strictly restrict downloaded media to confirmed image MIME types and image file extensions, discarding `.pdf` and tiny UI icons.

## Consequences
- **Positive**: Face detection success rate on crawled sites improved dramatically because RetinaFace operates on full-resolution source photos rather than micro-thumbnails.
- **Positive**: Crawler throughput and network bandwidth efficiency improved by eliminating PDF and icon noise.
- **Trade-off**: Downloading full-resolution images consumes more bandwidth per image, offset by concurrent batch fetching with `asyncio.Semaphore(12)`.
