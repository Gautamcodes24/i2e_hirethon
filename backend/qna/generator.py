"""LLM answer generator with citation-aware prompting.

Supports both Groq and OpenAI as LLM providers (configured via settings).
Constructs structured prompts with citation metadata so the LLM
can reference specific pages, sections, and content types.
"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


def _detect_pdf_name() -> str:
    """Find the PDF filename in the data directory."""
    for p in settings.DATA_DIR.iterdir():
        if p.suffix.lower() == ".pdf":
            return p.name
    return "document.pdf"


class AnswerGenerator:
    """Generate answers from retrieved chunks using the configured LLM provider."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        """Lazy-load the LLM client via the unified provider factory."""
        if self._client is None:
            from backend.utils.llm_client import get_llm_client
            self._client = get_llm_client()
        return self._client

    def generate(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a grounded answer with citations.

        Parameters
        ----------
        query : str
            The user's question.
        results : list
            Retrieval results from HybridRetriever.search().

        Returns
        -------
        dict with 'answer', 'citations', 'source_chunks'
        """
        if not results:
            return {"answer": "No relevant content found for your question.", "citations": [], "source_chunks": []}

        # Build context with citation markers
        context_lines = []
        citations = []
        for i, r in enumerate(results, 1):
            chunk = r["chunk"]
            # Use parent content if available (richer context)
            content = r.get("parent_content", chunk.get("content", ""))
            page = chunk.get("page", "?")
            section = chunk.get("section_heading", "")
            ctype = chunk.get("type", "text")
            caption = chunk.get("metadata", {}).get("caption_label", "")

            # Build citation label
            cite_parts = [f"Page {page}"]
            if section:
                cite_parts.append(section)
            cite_parts.append(ctype)
            if caption:
                cite_parts.append(caption)
            cite_label = ", ".join(cite_parts)
            citations.append({
                "index": i,
                "citation": cite_label,
                "score": r.get("score", 0),
                "page": page,
                "pdf_name": _detect_pdf_name(),
            })

            context_lines.append(f"[Source {i}: {cite_label}]\n{content.strip()}")

        context_text = "\n\n---\n\n".join(context_lines)

        prompt = self._build_prompt(query, context_text)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=settings.active_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert technical assistant. Answer questions accurately "
                            "using ONLY the provided context. Cite sources using [Source N] format. "
                            "If the context doesn't contain enough information, say so clearly. "
                            "Be concise and precise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_completion_tokens=settings.LLM_MAX_TOKENS,
            )

            answer = "I couldn't generate an answer."
            if hasattr(response, "choices") and response.choices:
                answer = response.choices[0].message.content.strip() or answer

            logger.info("Generated answer: %d chars, %d citations", len(answer), len(citations))

            return {
                "answer": answer,
                "citations": citations,
                "source_chunks": [r["chunk"] for r in results],
            }

        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            return {
                "answer": f"Error generating answer: {exc}",
                "citations": citations,
                "source_chunks": [r["chunk"] for r in results],
            }

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            f"Context (retrieved from the document):\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Instructions: Answer the question using the context above. "
            f"Cite your sources using [Source N] format. "
            f"If multiple sources support your answer, cite all of them."
        )

    def generate_stream(
        self,
        query: str,
        results: list[dict[str, Any]],
    ):
        """Yield answer tokens as they arrive (SSE-friendly generator).

        Yields dicts: {"type": "token", "content": "..."} for each chunk,
        then {"type": "citations", "citations": [...]} at the end.
        """
        if not results:
            yield {"type": "token", "content": "No relevant content found for your question."}
            yield {"type": "done"}
            return

        context_lines = []
        citations = []
        for i, r in enumerate(results, 1):
            chunk = r["chunk"]
            content = r.get("parent_content", chunk.get("content", ""))
            page = chunk.get("page", "?")
            section = chunk.get("section_heading", "")
            ctype = chunk.get("type", "text")
            caption = chunk.get("metadata", {}).get("caption_label", "")
            cite_parts = [f"Page {page}"]
            if section:
                cite_parts.append(section)
            cite_parts.append(ctype)
            if caption:
                cite_parts.append(caption)
            cite_label = ", ".join(cite_parts)
            citations.append({
                "index": i,
                "citation": cite_label,
                "score": r.get("score", 0),
                "page": page,
                "pdf_name": _detect_pdf_name(),
            })
            context_lines.append(f"[Source {i}: {cite_label}]\n{content.strip()}")

        context_text = "\n\n---\n\n".join(context_lines)
        prompt = self._build_prompt(query, context_text)

        try:
            client = self._get_client()
            stream = client.chat.completions.create(
                model=settings.active_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert technical assistant. Answer questions accurately "
                            "using ONLY the provided context. Cite sources using [Source N] format. "
                            "If the context doesn't contain enough information, say so clearly. "
                            "Be concise and precise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_completion_tokens=settings.LLM_MAX_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {"type": "token", "content": chunk.choices[0].delta.content}

            yield {"type": "citations", "citations": citations}
            yield {"type": "done"}

        except Exception as exc:
            logger.exception("LLM streaming failed: %s", exc)
            yield {"type": "error", "content": str(exc)}
