"""QnA pipeline — retrieval + generation orchestrator.

Single interface that combines hybrid retrieval and LLM answer generation.

Usage:
    from backend.qna import QnAPipeline

    pipeline = QnAPipeline()
    result = pipeline.ask("What is systems engineering?")
    print(result["answer"])
    print(result["citations"])
"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.logger import get_logger
from backend.qna.generator import AnswerGenerator
from backend.qna.retriever import HybridRetriever

logger = get_logger(__name__)


class QnAPipeline:
    """Orchestrates retrieval + generation with a clean interface."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

    @property
    def is_ready(self) -> bool:
        """Check if retrieval artefacts are loaded and ready."""
        self.retriever.load()
        return self.retriever.is_loaded

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        alpha: float | None = None,
        use_parent: bool | None = None,
    ) -> dict[str, Any]:
        """Ask a question and get a grounded answer with citations.

        Parameters
        ----------
        question : str
            The user's question.
        top_k : int, optional
            Number of chunks to retrieve (default from config).
        alpha : float, optional
            Dense/sparse weight (default from config).
        use_parent : bool, optional
            Whether to resolve parent chunks (default from config).

        Returns
        -------
        dict with keys: question, answer, citations, source_chunks, scores
        """
        logger.info("Question: %s", question[:100])

        # Step 1: Retrieve
        results = self.retriever.search(
            question,
            k=top_k,
            alpha=alpha,
            use_parent=use_parent,
        )
        logger.info("Retrieved %d results", len(results))

        if not results:
            return {
                "question": question,
                "answer": "No relevant content found. Have you ingested a PDF?",
                "citations": [],
                "source_chunks": [],
                "scores": [],
            }

        # Step 2: Generate answer
        gen_result = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "source_chunks": gen_result["source_chunks"],
            "scores": [r["score"] for r in results],
        }
