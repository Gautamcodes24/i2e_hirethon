"""Figure and table caption detection and linking.

Detects caption patterns such as:
  - "Figure 2.3-1: Systems Engineering Engine"
  - "Fig. 5: Overview of Process"
  - "Table 4.2-1: Life Cycle Phases"
  - "Exhibit A-1: Budget Summary"

Links detected captions to nearby extracted images and tables by page
proximity and position, enriching chunk metadata for better retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ── Caption regex patterns ────────────────────────────────────────────────────
_FIGURE_CAPTION = re.compile(
    r"(?:^|\n)\s*"
    r"(?:Figure|Fig\.?|FIGURE)\s+"
    r"([\d]+(?:[.\-][\d]+)*)"          # figure number: 2.3-1, 5, A-1
    r"[:\.\s\-–—]+"                    # separator
    r"([^\n]{5,200})",                 # caption text (5-200 chars)
    re.IGNORECASE | re.MULTILINE,
)

_TABLE_CAPTION = re.compile(
    r"(?:^|\n)\s*"
    r"(?:Table|TABLE|Tbl\.?)\s+"
    r"([\d]+(?:[.\-][\d]+)*)"          # table number
    r"[:\.\s\-–—]+"
    r"([^\n]{5,200})",
    re.IGNORECASE | re.MULTILINE,
)

_EXHIBIT_CAPTION = re.compile(
    r"(?:^|\n)\s*"
    r"(?:Exhibit|EXHIBIT)\s+"
    r"([A-Z]?[\d]+(?:[.\-][\d]+)*)"
    r"[:\.\s\-–—]+"
    r"([^\n]{5,200})",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Caption:
    """A detected caption in the document."""
    kind: str           # "figure", "table", "exhibit"
    number: str         # e.g. "2.3-1"
    text: str           # caption text
    page: int
    char_offset: int    # character position in page text

    @property
    def label(self) -> str:
        kind_map = {"figure": "Figure", "table": "Table", "exhibit": "Exhibit"}
        return f"{kind_map.get(self.kind, self.kind.title())} {self.number}"

    @property
    def full_label(self) -> str:
        return f"{self.label}: {self.text}"


# ── Public API ────────────────────────────────────────────────────────────────

def extract_captions(page_text: str, page_num: int) -> list[Caption]:
    """Extract all figure/table/exhibit captions from page text."""
    captions: list[Caption] = []
    seen_offsets: set[int] = set()

    for pattern, kind in [
        (_FIGURE_CAPTION, "figure"),
        (_TABLE_CAPTION, "table"),
        (_EXHIBIT_CAPTION, "exhibit"),
    ]:
        for m in pattern.finditer(page_text):
            offset = m.start()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            number = m.group(1).strip()
            text = m.group(2).strip()
            # Clean trailing punctuation artefacts
            text = re.sub(r"[.\s]+$", "", text)

            captions.append(Caption(
                kind=kind,
                number=number,
                text=text,
                page=page_num,
                char_offset=offset,
            ))

    return sorted(captions, key=lambda c: c.char_offset)


def link_captions_to_images(
    captions: list[Caption],
    images: list[dict[str, Any]],
) -> None:
    """Link figure captions to extracted images by page proximity.

    Mutates image dicts in-place, adding ``caption``, ``caption_label``,
    and ``figure_number`` keys.
    """
    figure_captions = [c for c in captions if c.kind == "figure"]
    if not figure_captions or not images:
        return

    for img in images:
        img_page = img.get("page", 0)
        # Find closest figure caption on same page or adjacent page
        best_cap: Caption | None = None
        best_dist = float("inf")

        for cap in figure_captions:
            dist = abs(cap.page - img_page)
            if dist < best_dist:
                best_dist = dist
                best_cap = cap
            elif dist == best_dist and best_cap and cap.char_offset < best_cap.char_offset:
                best_cap = cap  # prefer earlier on same page

        if best_cap and best_dist <= 1:  # same page or adjacent
            img["caption"] = best_cap.text
            img["caption_label"] = best_cap.full_label
            img["figure_number"] = best_cap.number


def link_captions_to_tables(
    captions: list[Caption],
    tables: list[dict[str, Any]],
) -> None:
    """Link table captions to detected tables by page proximity.

    Mutates table dicts in-place, adding ``caption``, ``caption_label``,
    and ``table_number`` keys.
    """
    table_captions = [c for c in captions if c.kind == "table"]
    if not table_captions or not tables:
        return

    for tbl in tables:
        tbl_pages = tbl.get("pages", [])
        if not tbl_pages:
            continue
        first_page = tbl_pages[0]

        best_cap: Caption | None = None
        best_dist = float("inf")

        for cap in table_captions:
            dist = abs(cap.page - first_page)
            if dist < best_dist:
                best_dist = dist
                best_cap = cap

        if best_cap and best_dist <= 1:
            tbl["caption"] = best_cap.text
            tbl["caption_label"] = best_cap.full_label
            tbl["table_number"] = best_cap.number


def captions_to_list(captions: list[Caption]) -> list[dict[str, Any]]:
    """Serialize captions to plain dicts."""
    return [
        {
            "kind": c.kind,
            "number": c.number,
            "text": c.text,
            "page": c.page,
            "label": c.label,
            "full_label": c.full_label,
        }
        for c in captions
    ]
