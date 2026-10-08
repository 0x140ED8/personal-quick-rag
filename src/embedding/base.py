"""向量化抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list) -> list:
        """批量编码文档块（不加指令）。"""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list:
        """编码查询（加指令）。"""
        raise NotImplementedError
