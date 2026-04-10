from fastapi.testclient import TestClient

from app.api import app, cache

client = TestClient(app)


def setup_function() -> None:
    cache.clear()


def test_query_endpoint_returns_answer() -> None:
    response = client.post("/query", json={"question": "How does caching help a RAG system?", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "sources" in payload
    assert payload["cached"] is False


def test_query_endpoint_uses_cache_on_second_request() -> None:
    client.post("/query", json={"question": "How does caching help a RAG system?", "top_k": 2})
    response = client.post("/query", json={"question": "How does caching help a RAG system?", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["cached"] is True
