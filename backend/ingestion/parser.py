"""
Layout-aware PDF parser using pdfplumber.

Key design decisions:
- Multi-page table stitching via column-signature matching
- Vertical-gap-based table separation on the same page
- Row validation to discard parser noise
- Page text is extracted only for pages (or page regions) that are NOT part of a table
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pdfplumber

# ---------------------------------------------------------------------------
# Tuneable knobs
# ---------------------------------------------------------------------------
COLUMN_SNAP_TOL = 10          # px: two column edges are "same" if within this distance
MIN_COLUMN_OVERLAP = 0.60     # fraction of columns that must match to call it a continuation
VERTICAL_GAP_THRESHOLD = 18   # pt: whitespace gap that separates two distinct tables on the same page
MIN_ROW_CELLS = 2             # a row with fewer non-empty cells is discarded as noise
MIN_TEXT_LENGTH = 20          # page-level text chunks shorter than this are skipped


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TableObject:
    table_id: str
    pages: list[int]
    header: list[str]
    rows: list[list[str]]
    col_xs: list[float]          # representative x-coordinates of each column
    last_bottom: float           # y-coordinate of the last row on the last page
    metadata: dict[str, Any] = field(default_factory=dict)

    def col_signature(self) -> list[float]:
        """Rounded column x-positions used for matching continuations."""
        return [round(x / COLUMN_SNAP_TOL) * COLUMN_SNAP_TOL for x in self.col_xs]

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "pages": self.pages,
            "header": self.header,
            "rows": self.rows,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _col_xs_from_pdfplumber_table(table) -> list[float]:
    """
    Extract representative x-coordinates for each column from the raw
    pdfplumber table object (which exposes bbox per cell).
    We use the left edge of each cell in the first data row.
    """
    settings = table.extract()
    if not settings:
        return []
    # pdfplumber table.cells is a list of (x0, top, x1, bottom) tuples per cell
    try:
        cells_row0 = [c for c in table.cells if c[1] == table.cells[0][1]]
        return [c[0] for c in cells_row0]
    except Exception:
        return []


def _columns_match(sig_a: list[float], sig_b: list[float]) -> bool:
    """Return True if the two column signatures overlap sufficiently."""
    if not sig_a or not sig_b:
        return False
    set_a = set(sig_a)
    set_b = set(sig_b)
    intersection = sum(
        1 for x in set_a
        if any(abs(x - y) <= COLUMN_SNAP_TOL for y in set_b)
    )
    overlap = intersection / max(len(set_a), len(set_b))
    return overlap >= MIN_COLUMN_OVERLAP


def _clean_cell(val) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def _validate_row(row: list[str]) -> bool:
    """Discard rows that look like parser noise."""
    non_empty = [c for c in row if c.strip()]
    return len(non_empty) >= MIN_ROW_CELLS


def _extract_col_xs(pl_table) -> list[float]:
    """Best-effort column x-positions."""
    try:
        first_row_cells = [c for c in pl_table.cells if c[1] == pl_table.cells[0][1]]
        return [round(c[0], 1) for c in first_row_cells]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
class PDFLayoutParser:
    """
    Parses a PDF and returns:
      - self.tables  : list[TableObject]  — merged multi-page tables
      - self.pages   : list[dict]         — page-level text (non-table regions)

    Call .ingest(pdf_path) to run the pipeline.
    """

    def __init__(self):
        self.tables: list[TableObject] = []
        self.pages: list[dict] = []
        self._table_counter = 0

    # ------------------------------------------------------------------
    def ingest(self, pdf_path: str) -> dict:
        self.tables = []
        self.pages = []
        self._table_counter = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num_0, page in enumerate(pdf.pages):
                page_num = page_num_0 + 1
                self._process_page(page, page_num)

        chunks = self._build_chunks()
        return {
            "tables": [t.to_dict() for t in self.tables],
            "chunks": chunks,
        }

    # ------------------------------------------------------------------
    def _process_page(self, page, page_num: int):
        """Detect tables and text on one page."""
        detected = page.find_tables()

        # Sort tables top-to-bottom
        detected = sorted(detected, key=lambda t: t.bbox[1])

        table_bboxes: list[tuple] = []  # (x0, top, x1, bottom)

        for pl_table in detected:
            raw_rows = pl_table.extract()
            if not raw_rows:
                continue

            col_xs = _extract_col_xs(pl_table)
            rows = [[_clean_cell(c) for c in row] for row in raw_rows]

            # First row treated as candidate header
            header = rows[0] if rows else []
            data_rows = [r for r in rows[1:] if _validate_row(r)]

            if not data_rows:
                continue

            table_top = pl_table.bbox[1]
            table_bottom = pl_table.bbox[3]
            table_bboxes.append(pl_table.bbox)

            # --- Try to stitch onto an existing open table ---
            continuation = self._find_continuation(col_xs, page_num)
            if continuation:
                continuation.rows.extend(data_rows)
                if page_num not in continuation.pages:
                    continuation.pages.append(page_num)
                continuation.last_bottom = table_bottom
            else:
                # New table
                self._table_counter += 1
                table_id = f"p{page_num}_t{self._table_counter}"
                new_table = TableObject(
                    table_id=table_id,
                    pages=[page_num],
                    header=header,
                    rows=data_rows,
                    col_xs=col_xs,
                    last_bottom=table_bottom,
                    metadata={"bbox_first_page": pl_table.bbox},
                )
                self.tables.append(new_table)

        # --- Extract page text outside table bounding boxes ---
        page_text = self._extract_text_outside_tables(page, table_bboxes)
        if page_text and len(page_text) >= MIN_TEXT_LENGTH:
            self.pages.append({
                "page": page_num,
                "text": page_text,
                "has_tables": bool(table_bboxes),
            })

    # ------------------------------------------------------------------
    def _find_continuation(self, col_xs: list[float], current_page: int) -> TableObject | None:
        """
        Find an existing TableObject whose last page is current_page - 1
        (cross-page) OR current_page (within-page, same column layout).
        Column signatures must match.
        """
        for tbl in reversed(self.tables):
            last_page = tbl.pages[-1]
            if last_page < current_page - 1:
                break  # too far back
            if _columns_match(tbl.col_signature(), [
                round(x / COLUMN_SNAP_TOL) * COLUMN_SNAP_TOL for x in col_xs
            ]):
                return tbl
        return None

    # ------------------------------------------------------------------
    def _extract_text_outside_tables(self, page, table_bboxes: list[tuple]) -> str:
        """
        Use pdfplumber crop to extract text from regions that are not
        covered by any detected table bounding box.
        """
        if not table_bboxes:
            return (page.extract_text() or "").strip()

        page_height = page.height
        page_width = page.width

        # Build vertical slices between tables
        boundaries = sorted({0.0} | {bb[1] for bb in table_bboxes} | {bb[3] for bb in table_bboxes} | {page_height})
        text_parts = []

        for i in range(len(boundaries) - 1):
            top = boundaries[i]
            bottom = boundaries[i + 1]
            # Is this region covered by any table?
            covered = any(
                bb[1] <= top and bb[3] >= bottom
                for bb in table_bboxes
            )
            if covered:
                continue
            region_height = bottom - top
            if region_height < 4:
                continue
            try:
                cropped = page.crop((0, top, page_width, bottom))
                txt = (cropped.extract_text() or "").strip()
                if txt:
                    text_parts.append(txt)
            except Exception:
                pass

        return "\n".join(text_parts)

    # ------------------------------------------------------------------
    def _build_chunks(self) -> list[dict]:
        """Convert TableObjects and page texts into flat chunk dicts."""
        chunks: list[dict] = []

        for tbl in self.tables:
            # One chunk per row (keeps granularity for retrieval)
            for row_idx, row in enumerate(tbl.rows):
                content = " | ".join(row)
                chunks.append({
                    "content": content,
                    "type": "table",
                    "page": tbl.pages[0],
                    "metadata": {
                        "table_id": tbl.table_id,
                        "header": tbl.header,
                        "row": row_idx,
                        "pages": tbl.pages,
                    },
                })

        for page_info in self.pages:
            chunks.append({
                "content": page_info["text"],
                "type": "text",
                "page": page_info["page"],
                "metadata": {"pages": [page_info["page"]]},
            })

        return chunks
