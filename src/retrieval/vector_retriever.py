"""向量（稠密）召回：query 编码成向量后交给向量库检索。"""
from __future__ import annotations

from src.retrieval.base import BaseRetriever
from src.types import Hit


class VectorRetriever(BaseRetriever):
    def __init__(self, store, embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        qvec = self._embedder.embed_query(query)
        return self._store.search_by_vector(qvec, top_k)
