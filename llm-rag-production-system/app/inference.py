"""Inference layer for the RAG service.

This module provides a mock LLM interface. In a production environment, this is
where you would integrate with OpenAI or another inference backend.
"""

from __future__ import annotations

from typing import Iterable

from app.retrieval import RetrievedDocument


class MockLLM:
    def generate(self, question: str, context_documents: Iterable[RetrievedDocument]) -> str:
        contexts = list(context_documents)
        if not contexts:
            return (
                "I could not find relevant context in the indexed corpus. "
                "In production, this response could trigger a fallback workflow "
                "or ask the user a clarifying question."
            )

        joined_context = " ".join(doc.text for doc in contexts[:2])
        return (
            f"Based on the retrieved context, the answer to '{question}' is: "
            f"{joined_context}"
        )
