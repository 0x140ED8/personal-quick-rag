"""BM25 关键词（稀疏）召回：rank-bm25 + jieba 分词。

chunk 文本通过 chunks.jsonl 持久化，加载时重建 BM25 索引（无需 pickle）。
"""
from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever
from src.types import Chunk, Hit


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks_file: str = None) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None
        self._chunks_file = chunks_file

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def save_chunks(self, chunks: list[Chunk], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        self._chunks_file = path

    def load(self) -> None:
        if not self._chunks_file or not Path(self._chunks_file).exists():
            return
        chunks = []
        with open(self._chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        self.build(chunks)

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for i in idxs:
            c = self._chunks[i]
            hits.append(
                Hit(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=float(scores[i]),
                    source=c.source,
                    heading_path=list(c.heading_path),
                    page=c.page,
                )
            )
        return hits

    @staticmethod
    def _tokenize(text: str) -> list:
        return [t for t in jieba.lcut(text.lower()) if t.strip()]
