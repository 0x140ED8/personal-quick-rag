"""向量存储抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Chunk, Hit


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[Chunk], embedder) -> None:
        """入库：向量化 Chunk 并写入索引。"""
        raise NotImplementedError

    @abstractmethod
    def from_vectors(self, vectors, chunks: list[Chunk]) -> None:
        """用预算好的向量整体重建索引（增量 ingest：缓存向量 + 新向量拼接）。"""
        raise NotImplementedError

    @abstractmethod
    def search_by_vector(self, vector: list, top_k: int) -> list[Hit]:
        """按向量召回 top_k 条，返回带分数的 Hit。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path, embedder) -> None:
        raise NotImplementedError
