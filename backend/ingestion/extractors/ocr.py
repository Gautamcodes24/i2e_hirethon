"""OCR helpers for scanned / image-only PDF pages.

Uses PyMuPDF (fitz) for page rendering and pytesseract for text recognition.
No external poppler dependency required — fitz renders pages natively.
"""

from __future__ import annotations

import logging
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum native-text character count below which a page is considered scanned
_MIN_NATIVE_CHARS = 20


def is_scanned_page(page: fitz.Page, min_chars: int = _MIN_NATIVE_CHARS) -> bool:
    """Return True if *page* appears to be a scanned image rather than native text.

    Heuristic: the page has fewer than *min_chars* extractable characters **and**
    contains at least one embedded image.
    """
    native_text = (page.get_text("text") or "").strip()
    has_images = bool(page.get_images(full=True))
    return len(native_text) < min_chars and has_images


def _page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image at the given DPI."""
    zoom = dpi / 72  # fitz default resolution is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(BytesIO(pix.tobytes("png")))


def ocr_page(page: fitz.Page, dpi: int = 300, lang: str = "eng") -> str:
    """Run Tesseract OCR on *page* and return the recognised text.

    Parameters
    ----------
    page : fitz.Page
        A PyMuPDF page object.
    dpi : int
        Resolution for rendering.  Higher → better OCR accuracy but slower.
    lang : str
        Tesseract language code (default ``"eng"``).

    Returns
    -------
    str
        The OCR-recognised text, stripped of leading/trailing whitespace.
        Returns an empty string on failure.
    """
    try:
        import pytesseract  # deferred so the rest of the module works without it
    except ImportError:
        logger.error(
            "pytesseract is not installed.  Install it with: pip install pytesseract  "
            "and ensure the Tesseract binary is on your PATH."
        )
        return ""

    try:
        img = _page_to_pil(page, dpi=dpi)
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except Exception:
        logger.exception("OCR failed for page %s", page.number + 1)
        return ""
