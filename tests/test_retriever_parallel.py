from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from src.retriever import ChromaRetriever


def test_hybrid_retrieval_runs_vector_and_keyword_searches_concurrently():
    retriever = object.__new__(ChromaRetriever)
    retriever.vectorstore = object()
    retriever._retrieval_executor = ThreadPoolExecutor(max_workers=2)
    barrier = Barrier(2)

    def vector_retrieve(query, top_k):
        barrier.wait(timeout=1)
        return [{"metadata": {"id": "vector"}, "text": "vector", "score": 1.0}]

    def keyword_retrieve(queries, top_k):
        barrier.wait(timeout=1)
        return [{"metadata": {"id": "keyword"}, "text": "keyword", "score": 1.0}]

    retriever._vector_retrieve = vector_retrieve
    retriever._keyword_retrieve_many = keyword_retrieve
    retriever._rrf_fuse = lambda vector, keyword, top_k: vector + keyword
    retriever._expand_parent_chunks = lambda chunks, top_k: chunks

    try:
        chunks = retriever.retrieve("test", top_k=1)
    finally:
        retriever._retrieval_executor.shutdown(wait=True)

    assert [chunk["metadata"]["id"] for chunk in chunks] == ["vector", "keyword"]
