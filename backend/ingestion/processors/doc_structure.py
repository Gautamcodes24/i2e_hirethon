"""Document structure analysis — TOC parsing and page classification.

Classifies each page into a structural category so the pipeline can:
  - Skip non-content pages (cover, blank, TOC itself)
  - Tag chunks with their document region (body, appendix, glossary, etc.)
  - Weight content pages higher than front/back matter in retrieval

Page categories:
  cover, title_page, toc, front_matter, body, appendix, glossary,
  index, references, blank, unknown
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Page classification patterns ──────────────────────────────────────────────
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "cover": [
        re.compile(r"(?:^|\n)\s*(?:cover\s*page|front\s*cover)\s*$", re.I | re.M),
    ],
    "title_page": [
        re.compile(
            r"(?:prepared\s+(?:by|for)|published\s+by|issued\s+by|"
            r"national\s+aeronautics|nasa\s+sp-|revision\s+\d)",
            re.I,
        ),
    ],
    "toc": [
        re.compile(r"(?:^|\n)\s*(?:TABLE OF CONTENTS|CONTENTS)\s*(?:\n|$)", re.I),
        # Dense dot-leader pattern (TOC specific)
        re.compile(r"(?:\.{4,}\s*\d+\s*\n){3,}"),
    ],
    "front_matter": [
        re.compile(r"(?:^|\n)\s*(?:PREFACE|FOREWORD|ACKNOWLEDGMENT|EXECUTIVE SUMMARY)\s*(?:\n|$)", re.I),
        re.compile(r"(?:^|\n)\s*(?:LIST OF FIGURES|LIST OF TABLES)\s*(?:\n|$)", re.I),
        re.compile(r"(?:^|\n)\s*(?:ABBREVIATIONS|ACRONYMS)\s*(?:\n|$)", re.I),
    ],
    "appendix": [
        re.compile(r"(?:^|\n)\s*APPENDIX\s+[A-Z]\b", re.I),
    ],
    "glossary": [
        re.compile(r"(?:^|\n)\s*(?:GLOSSARY|DEFINITIONS)\s*(?:\n|$)", re.I),
    ],
    "index_page": [
        re.compile(r"(?:^|\n)\s*(?:INDEX|SUBJECT INDEX)\s*(?:\n|$)", re.I),
    ],
    "references": [
        re.compile(r"(?:^|\n)\s*(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED)\s*(?:\n|$)", re.I),
    ],
}

# Minimum text length to consider a page non-blank
_MIN_TEXT_FOR_CONTENT = 50


@dataclass
class PageInfo:
    """Classification and metadata for a single page."""
    page_num: int
    category: str = "body"
    is_content: bool = True         # should this page be included in chunks?
    text_length: int = 0
    has_tables: bool = False
    has_images: bool = False


@dataclass
class DocumentStructure:
    """Full document structure analysis."""
    pages: list[PageInfo] = field(default_factory=list)
    toc_entries: list[dict[str, Any]] = field(default_factory=list)
    total_pages: int = 0
    body_start: int = 1             # first body-content page
    appendix_start: int | None = None

    def page_info(self, page_num: int) -> PageInfo | None:
        for p in self.pages:
            if p.page_num == page_num:
                return p
        return None

    def is_content_page(self, page_num: int) -> bool:
        info = self.page_info(page_num)
        return info.is_content if info else True

    def category_for(self, page_num: int) -> str:
        info = self.page_info(page_num)
        return info.category if info else "body"

    def summary(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(p.category for p in self.pages))


# ── Public API ────────────────────────────────────────────────────────────────

def classify_page(text: str, page_num: int, total_pages: int) -> str:
    """Classify a single page into a structural category."""
    stripped = text.strip()

    # Blank page
    if len(stripped) < _MIN_TEXT_FOR_CONTENT:
        return "blank"

    # First few pages are likely cover/title
    if page_num <= 2 and len(stripped) < 500:
        for pattern in _PATTERNS.get("cover", []):
            if pattern.search(stripped):
                return "cover"
        # Short first pages are likely title pages
        if page_num == 1:
            return "title_page"

    # Check each category pattern
    for category, patterns in _PATTERNS.items():
        for pat in patterns:
            if pat.search(stripped):
                return category

    return "body"


def analyze_document_structure(
    page_texts: list[str],
    *,
    skip_non_content: bool = True,
) -> DocumentStructure:
    """Analyze the full document and classify every page.

    Parameters
    ----------
    page_texts : list of str
        Text extracted from each page (index 0 = page 1).
    skip_non_content : bool
        If True, mark non-content pages (cover, blank, TOC) as not for chunking.

    Returns
    -------
    DocumentStructure
    """
    total = len(page_texts)
    structure = DocumentStructure(total_pages=total)

    # Phase 1: classify each page
    for idx, text in enumerate(page_texts):
        page_num = idx + 1
        category = classify_page(text, page_num, total)

        # Non-content categories
        non_content = {"cover", "title_page", "toc", "blank"}
        is_content = category not in non_content if skip_non_content else True

        structure.pages.append(PageInfo(
            page_num=page_num,
            category=category,
            is_content=is_content,
            text_length=len(text.strip()),
        ))

    # Phase 2: find body start (first "body" page after front matter)
    for p in structure.pages:
        if p.category == "body":
            structure.body_start = p.page_num
            break

    # Phase 3: find appendix start
    for p in structure.pages:
        if p.category == "appendix":
            structure.appendix_start = p.page_num
            break

    # Phase 4: parse TOC entries if TOC pages exist
    toc_pages = [p for p in structure.pages if p.category == "toc"]
    if toc_pages:
        for p in toc_pages:
            idx = p.page_num - 1
            if idx < len(page_texts):
                structure.toc_entries.extend(
                    parse_toc_entries(page_texts[idx], p.page_num)
                )

    return structure


def parse_toc_entries(toc_text: str, toc_page: int) -> list[dict[str, Any]]:
    """Extract TOC entries (section title → page number) from TOC text.

    Returns list of dicts with ``title``, ``target_page``, ``level``.
    """
    entries: list[dict[str, Any]] = []

    # Pattern: "2.3 Section Title ........... 42" or "Section Title    42"
    toc_line = re.compile(
        r"^"
        r"(?:(\d+(?:\.\d+)*)\s+)?"          # optional section number
        r"(.+?)"                              # title
        r"\s*[.\s]{3,}\s*"                   # dot leaders or whitespace
        r"(\d{1,4})"                          # page number
        r"\s*$",
        re.MULTILINE,
    )

    for m in toc_line.finditer(toc_text):
        number = (m.group(1) or "").strip()
        title = m.group(2).strip()
        target_page = int(m.group(3))

        # Heuristic: determine nesting level from number
        level = number.count(".") + 1 if number else 0

        if title and len(title) > 2:
            entries.append({
                "number": number,
                "title": title,
                "target_page": target_page,
                "level": level,
                "toc_page": toc_page,
            })

    return entries
