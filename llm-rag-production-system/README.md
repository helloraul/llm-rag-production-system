# llm-rag-production-system

Production-ready RAG system for low-latency question answering over large document corpora.

This project is intentionally simple in implementation but strong in system design. It demonstrates how to structure a Retrieval-Augmented Generation service as a production-minded backend rather than a one-off script.

## What this repo demonstrates

- End-to-end RAG flow: query → retrieval → LLM → response
- Clean separation of concerns across API, retrieval, inference, and caching
- Production-oriented thinking: observability, rate limiting hooks, fault handling, and scaling notes
- A design that can evolve from a local demo into a horizontally scalable service

## Problem statement

Teams often need low-latency question answering over internal knowledge bases, policy documents, runbooks, or product documentation. A production RAG system needs more than retrieval and generation: it also needs cacheability, clear interfaces, resilience, and operational visibility.

This repository shows one way to build that system.

## Architecture

High-level request flow:

1. A client sends a question to the `/query` endpoint.
2. The API normalizes the request and checks the cache.
3. On a cache miss, the retrieval layer searches the local corpus for the most relevant documents.
4. The inference layer builds a prompt from retrieved context and generates an answer.
5. The response is cached and returned with metadata.

### Request path

```text
Client Request
   ↓
FastAPI /query
   ↓
Cache lookup
   ↓ (miss)
Retriever
   ↓
Prompt assembly
   ↓
LLM inference
   ↓
Cache write
   ↓
Response
```

### Components

- `app/api.py`: FastAPI application and `/query` endpoint
- `app/retrieval.py`: document loading and simple vector-style retrieval
- `app/inference.py`: mock LLM interface and prompt assembly
- `app/cache.py`: in-memory TTL cache with a Redis-like interface
- `app/main.py`: local entrypoint

## Tradeoffs

- **Used simple vector search instead of FAISS in code** to keep the project lightweight and runnable anywhere. In a production deployment, FAISS is a strong choice for low-latency dense retrieval at scale.
- **Implemented caching** to reduce repeated inference cost and improve latency for frequent or duplicate queries.
- **Used a mock LLM wrapper** so the service remains self-contained. This makes the system easy to run locally while preserving a clear integration point for OpenAI or another model provider.
- **Kept retrieval local and file-based** for portability. In production, the corpus would typically live in object storage, a document pipeline, or an indexed vector store.
- **Discussed CPU vs GPU inference tradeoffs** rather than coupling the sample to a GPU runtime. For low traffic or smaller models, CPU may be sufficient; for higher throughput or larger models, GPU-backed inference is often required.

## Scaling considerations

- **Horizontal scaling of the API layer**: the FastAPI service is stateless except for local cache and document index state, so it can be replicated behind a load balancer.
- **Batching for LLM calls**: in production, the inference layer can batch multiple requests to improve throughput and reduce per-request overhead.
- **Async request handling**: FastAPI supports asynchronous endpoints, which helps when inference or network calls are I/O bound.
- **External cache**: replace the in-memory cache with Redis or Memcached for shared cache state across replicas.
- **External retrieval index**: replace the local retriever with FAISS, a hosted vector database, or a hybrid search system.

## Production considerations

### Observability

- Structured logging for requests, cache hits, and latency
- Metrics for request count, p50/p95 latency, cache hit rate, and retrieval time
- Tracing across API, retrieval, inference, and downstream model calls

### Rate limiting

- Add per-tenant or per-key request limits at the API gateway or application layer
- Protect expensive inference paths from abuse or accidental spikes

### Fault tolerance

- Fallback responses when retrieval returns no relevant context
- Timeout budgets for inference calls
- Retries with backoff for transient downstream errors
- Circuit breakers around external model or vector-store dependencies

## CPU vs GPU inference tradeoffs

- **CPU**: lower cost and simpler operations, suitable for lower throughput or smaller models
- **GPU**: better latency and throughput for larger models and high-concurrency traffic, but requires capacity planning and autoscaling strategy
- **Hybrid**: route smaller or cached requests through CPU paths and reserve GPU for heavy or latency-sensitive generation

## Local development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

The API will start on `http://127.0.0.1:8000`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does caching help a RAG system?","top_k":3}'
```

### Example response

```json
{
  "answer": "Caching reduces repeated retrieval and inference work, which lowers latency and cost.",
  "sources": [
    {
      "document_id": "doc_2",
      "score": 0.61,
      "text": "Caching can reduce repeated LLM calls and improve latency for repeated questions."
    }
  ],
  "cached": false,
  "latency_ms": 12.4
}
```

## Future improvements

- Swap in FAISS or a hosted vector database
- Add an OpenAI API adapter in `inference.py`
- Add hybrid retrieval (BM25 + dense vectors)
- Add tenant-aware auth and rate limiting
- Add OpenTelemetry tracing and Prometheus metrics
- Add document ingestion and chunking pipeline

## Testing

Run tests with:

```bash
pytest -q
```

## Why this repo is useful in an interview

This repo is designed to show system design thinking in code:

- clear interfaces
- separation of concerns
- production-minded tradeoffs
- a path from local demo to scalable service

