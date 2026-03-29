"""QnA pipeline sub-package.

Modules for hybrid retrieval (BM25 + dense), parent-child resolution,
LLM answer generation with citations, and the query orchestrator.

Usage:
    from backend.qna import QnAPipeline

    pipeline = QnAPipeline()
    answer = pipeline.ask("What is systems engineering?")
"""

from backend.qna.pipeline import QnAPipeline

__all__ = ["QnAPipeline"]
