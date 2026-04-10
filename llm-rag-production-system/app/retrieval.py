"""Retrieval layer for the RAG service.

The implementation uses a lightweight token-frequency cosine similarity model
so the project stays easy to run locally. The class boundaries are intentionally
structured so this can later be replaced with FAISS or another vector store.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


@dataclass
class Document:
    document_id: str
    text: str


@dataclass
class RetrievedDocument:
    document_id: str
    text: str
    score: float


class SimpleRetriever:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self.doc_vectors = [self._vectorize(doc.text) for doc in self.documents]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    @classmethod
    def _vectorize(cls, text: str) -> Counter:
        return Counter(cls._tokenize(text))

    @staticmethod
    def _cosine_similarity(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0

        intersection = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in intersection)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        query_vector = self._vectorize(query)
        scored: List[RetrievedDocument] = []

        for doc, doc_vector in zip(self.documents, self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                scored.append(
                    RetrievedDocument(
                        document_id=doc.document_id,
                        text=doc.text,
                        score=round(score, 4),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def load_documents(data_path: Path) -> List[Document]:
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    documents: List[Document] = []
    for item in payload:
        documents.append(Document(document_id=item["document_id"], text=item["text"]))
    return documents
