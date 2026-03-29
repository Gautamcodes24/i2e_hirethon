"""Ingestion pipeline sub-package.

Modules for parsing PDFs, extracting content, chunking, cleaning,
scoring, deduplicating, and building vector indices.

Usage:
    from backend.ingestion import IngestPipeline

    pipeline = IngestPipeline()
    result = pipeline.run("path/to/doc.pdf")
"""

from backend.ingestion.pipeline import IngestPipeline

__all__ = ["IngestPipeline"]
