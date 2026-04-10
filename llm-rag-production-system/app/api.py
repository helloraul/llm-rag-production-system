"""FastAPI application for the RAG system."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.cache import TTLCache
from app.inference import MockLLM
from app.retrieval import RetrievedDocument, SimpleRetriever, load_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag-api")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "documents.json"

retriever = SimpleRetriever(load_documents(DATA_PATH))
llm = MockLLM()
cache = TTLCache(default_ttl_seconds=300)

app = FastAPI(title="LLM RAG Production System", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")


class SourceItem(BaseModel):
    document_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    cached: bool
    latency_ms: float


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    cache_key = f"question:{payload.question.lower().strip()}|top_k:{payload.top_k}"

    cached_response = cache.get(cache_key)
    if cached_response is not None:
        logger.info("cache_hit question=%s", payload.question)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        cached_response["latency_ms"] = elapsed_ms
        cached_response["cached"] = True
        return QueryResponse(**cached_response)

    logger.info("cache_miss question=%s", payload.question)
    try:
        results: List[RetrievedDocument] = retriever.search(payload.question, top_k=payload.top_k)
        answer = llm.generate(payload.question, results)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response_payload = {
        "answer": answer,
        "sources": [
            {"document_id": doc.document_id, "score": doc.score, "text": doc.text}
            for doc in results
        ],
        "cached": False,
        "latency_ms": elapsed_ms,
    }
    cache.set(cache_key, response_payload)
    return QueryResponse(**response_payload)
