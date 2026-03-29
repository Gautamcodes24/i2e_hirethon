"""Image / diagram extraction and vision-LLM description.

Extracts embedded images from PDF pages via PyMuPDF, filters out decorative
ones, and optionally sends them to a Vision LLM (Groq or OpenAI) for a
textual description suitable for embedding and retrieval.

Also detects **vector diagrams** (flowcharts, line drawings, arrows) that
are rendered as PDF path objects rather than raster images.  These are
invisible to ``get_images()`` but can be captured by rendering the page
and checking for drawing operations.
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# ── Filtering thresholds ──────────────────────────────────────────────────────
MIN_IMAGE_WIDTH = 100   # px
MIN_IMAGE_HEIGHT = 100  # px
MAX_IMAGES_PER_PAGE = 5
VISION_BATCH_DELAY = 1.0  # seconds between Groq vision API calls
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # exponential back-off base (seconds)

# ── Vision prompt ─────────────────────────────────────────────────────────────
_VISION_PROMPT = (
    "Describe this technical diagram, workflow, chart, or figure in detail "
    "for a question-answering system. Include all visible labels, arrows, "
    "relationships, data values, and any text present in the image. "
    "If it is a table rendered as an image, reproduce the table content. "
    "Be precise and factual."
)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_images(
    page: fitz.Page,
    page_num: int,
    doc: fitz.Document,
) -> list[dict[str, Any]]:
    """Extract embedded images from *page*, skipping decorative ones.

    Returns a list of dicts, each containing:
        image_bytes, image_b64, width, height, page, bbox, xref
    """
    results: list[dict[str, Any]] = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list[:MAX_IMAGES_PER_PAGE]):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            if not base_image or not base_image.get("image"):
                continue

            image_bytes: bytes = base_image["image"]
            pil_img = Image.open(BytesIO(image_bytes))
            width, height = pil_img.size

            # Skip decorative / tiny images (logos, icons, bullets)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            # Convert to PNG for consistent handling
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            results.append({
                "image_bytes": png_bytes,
                "image_b64": base64.b64encode(png_bytes).decode("ascii"),
                "width": width,
                "height": height,
                "page": page_num,
                "bbox": None,  # PyMuPDF get_images doesn't give bbox directly
                "xref": xref,
            })
        except Exception:
            logger.debug("Failed to extract image xref=%s on page %s", xref, page_num, exc_info=True)

    return results


def describe_image_with_vision(
    image_b64: str,
    groq_api_key: str,
    *,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    prompt: str = _VISION_PROMPT,
) -> str:
    """Send a base64 image to the configured Vision LLM and return its description.

    Supports both Groq and OpenAI via ``backend.utils.llm_client.get_vision_client``.
    Includes retry with exponential back-off for transient API errors.
    """
    from backend.config import settings
    from backend.utils.llm_client import get_vision_client

    try:
        client = get_vision_client()
    except Exception as exc:
        logger.error("Failed to create vision client: %s", exc)
        return ""

    # Use the active vision model from config
    active_model = settings.active_vision_model

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_completion_tokens=512,
            )
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content.strip()
            return ""
        except Exception as exc:
            wait = _RETRY_BACKOFF ** attempt
            logger.warning(
                "Vision API attempt %d/%d failed: %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, exc, wait,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    logger.error("Vision API failed after %d retries.", _MAX_RETRIES)
    return ""


def detect_vector_diagrams(
    page: fitz.Page,
    page_num: int,
    *,
    min_drawings: int = 5,
    min_raster_images: int = 0,
) -> list[dict[str, Any]]:
    """Detect pages with vector diagrams (flowcharts, line art, arrows).

    Vector diagrams are drawn using PDF path operations (lines, curves, rects)
    rather than embedded raster images.  This function renders such pages
    and returns them as "virtual" images for vision-LLM description.

    Parameters
    ----------
    page : fitz.Page
    page_num : int
    min_drawings : int
        Minimum number of drawing operations (paths) on the page to qualify.
    min_raster_images : int
        If the page already has this many raster images extracted, skip
        vector detection (assumes raster images cover the content).

    Returns
    -------
    list of image dicts (same schema as extract_images output)
    """
    results: list[dict[str, Any]] = []

    try:
        # Count drawing operations on this page
        drawings = page.get_drawings()
        n_drawings = len(drawings)

        if n_drawings < min_drawings:
            return []

        # Check if the page already has enough raster images
        n_raster = len(page.get_images(full=True))
        if n_raster > min_raster_images:
            return []

        # This page likely contains vector graphics — render it
        zoom = 2.0  # render at 144 dpi (2x default 72)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")

        pil_img = Image.open(BytesIO(png_bytes))
        width, height = pil_img.size

        results.append({
            "image_bytes": png_bytes,
            "image_b64": base64.b64encode(png_bytes).decode("ascii"),
            "width": width,
            "height": height,
            "page": page_num,
            "bbox": (0, 0, page.rect.width, page.rect.height),
            "xref": None,
            "is_vector": True,
            "n_drawings": n_drawings,
        })

        logger.info(
            "Vector diagram detected on page %d (%d drawing ops, %d raster images)",
            page_num, n_drawings, n_raster,
        )

    except Exception:
        logger.debug("Vector diagram detection failed on page %d", page_num, exc_info=True)

    return results


def describe_images_batch(
    images: list[dict[str, Any]],
    groq_api_key: str,
    *,
    delay: float = VISION_BATCH_DELAY,
) -> list[dict[str, Any]]:
    """Describe a batch of images, respecting rate limits.

    Mutates each image dict in-place by adding a ``description`` key and
    returns the same list for convenience.
    """
    for i, img in enumerate(images):
        if i > 0:
            time.sleep(delay)
        desc = describe_image_with_vision(img["image_b64"], groq_api_key)
        img["description"] = desc
        logger.info(
            "Described image %d/%d (page %s): %s",
            i + 1, len(images), img.get("page"), desc[:80] if desc else "<empty>",
        )
    return images
