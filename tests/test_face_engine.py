"""
Unit tests for FaceEngine preprocessing, blur detection, size validation, and cropping.
"""

import numpy as np
import pytest
from app.core.exceptions import InvalidImageError
from app.models.face import BoundingBox
from app.services.face_engine import FaceEngine


def test_decode_valid_image(synthetic_face_image_bytes: bytes):
    img = FaceEngine.decode_image(synthetic_face_image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape[0] == 200
    assert img.shape[1] == 200
    assert img.shape[2] == 3


def test_decode_invalid_image():
    with pytest.raises(InvalidImageError):
        FaceEngine.decode_image(b"not-an-image-data")


def test_blur_detection_sharp_vs_blurry(synthetic_face_image_bytes: bytes, blurry_image_bytes: bytes):
    sharp_img = FaceEngine.decode_image(synthetic_face_image_bytes)
    blurry_img = FaceEngine.decode_image(blurry_image_bytes)

    sharp_score = FaceEngine.compute_blur_score(sharp_img)
    blurry_score = FaceEngine.compute_blur_score(blurry_img)

    assert sharp_score > blurry_score
    assert blurry_score < 10.0


def test_quality_assessment_min_size(synthetic_face_image_bytes: bytes):
    engine = FaceEngine.get_instance()
    img = FaceEngine.decode_image(synthetic_face_image_bytes)

    # Valid large bbox
    valid_bbox = BoundingBox(x1=10, y1=10, x2=150, y2=150)
    report = engine.assess_quality(img, valid_bbox)
    assert report.meets_min_size is True
    assert report.face_width == 140
    assert report.face_height == 140

    # Small bbox 30px (meets relaxed 24px min size)
    small_bbox = BoundingBox(x1=10, y1=10, x2=40, y2=40)
    small_report = engine.assess_quality(img, small_bbox)
    assert small_report.meets_min_size is True

    # Tiny bbox < 24px
    tiny_bbox = BoundingBox(x1=10, y1=10, x2=25, y2=25)
    tiny_report = engine.assess_quality(img, tiny_bbox)
    assert tiny_report.meets_min_size is False
    assert tiny_report.passed is False


def test_crop_face(synthetic_face_image_bytes: bytes):
    img = FaceEngine.decode_image(synthetic_face_image_bytes)
    bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
    cropped = FaceEngine.crop_face(img, bbox, padding=0.0)
    assert cropped.shape[0] == 100
    assert cropped.shape[1] == 100


def test_crawler_rich_html_extraction():
    from bs4 import BeautifulSoup
    from app.services.crawler import FullSiteCrawlerService

    html = """
    <html>
        <head>
            <title>Test Page</title>
            <meta property="og:image" content="https://example.com/meta_hero.jpg" />
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Person",
                "name": "Target Person",
                "image": "https://example.com/jsonld_profile.jpg"
            }
            </script>
        </head>
        <body>
            <div style="background-image: url('https://example.com/css_bg.png');">
                <picture>
                    <source srcset="https://example.com/picture_srcset.webp 1x, https://example.com/picture_srcset_2x.webp 2x">
                    <img data-src="https://example.com/lazy_photo.jpg" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" />
                </picture>
            </div>
            <a href="https://example.com/uploads/2024/gallery_full.jpg">View Full Image</a>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    images = FullSiteCrawlerService._extract_images_from_soup(soup, "https://example.com/article")

    assert "https://example.com/meta_hero.jpg" in images
    assert "https://example.com/jsonld_profile.jpg" in images
    assert "https://example.com/css_bg.png" in images
    assert "https://example.com/picture_srcset.webp" in images
    assert "https://example.com/picture_srcset_2x.webp" in images
    assert "https://example.com/lazy_photo.jpg" in images
    assert "https://example.com/uploads/2024/gallery_full.jpg" in images

