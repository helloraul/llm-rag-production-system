from app.retrieval import Document, SimpleRetriever


def test_retriever_returns_relevant_document() -> None:
    retriever = SimpleRetriever(
        [
            Document(document_id="1", text="Caching improves latency for repeated requests."),
            Document(document_id="2", text="Kubernetes helps schedule containers."),
        ]
    )

    results = retriever.search("How does caching reduce latency?", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == "1"
