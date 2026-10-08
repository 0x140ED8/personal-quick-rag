"""双路召回 + RRF 融合：向量召回与 BM25 召回各自取 top_k，再按倒数排名融合。"""
from __future__ import annotations

from src.retrieval.base import BaseRetriever
from src.types import Hit


def rrf_fuse(hits_a: list[Hit], hits_b: list[Hit], k: int = 60) -> list[Hit]:
    rrf: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for rank, hit in enumerate(hits_a):
        by_id.setdefault(hit.chunk_id, hit)
        rrf[hit.chunk_id] = rrf.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, hit in enumerate(hits_b):
        by_id.setdefault(hit.chunk_id, hit)
        rrf[hit.chunk_id] = rrf.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    out = []
    for cid, score in ranked:
        h = by_id[cid]
        out.append(
            Hit(
                chunk_id=cid,
                text=h.text,
                score=score,
                source=h.source,
                heading_path=h.heading_path,
                page=h.page,
            )
        )
    return out


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense: BaseRetriever,
        sparse: BaseRetriever,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        dense_hits = self._dense.retrieve(query, self._dense_top_k)
        sparse_hits = self._sparse.retrieve(query, self._sparse_top_k)
        fused = rrf_fuse(dense_hits, sparse_hits, self._rrf_k)
        return fused[:top_k]
