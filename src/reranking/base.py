"""重排序抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Hit


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[Hit], top_k: int = None) -> list[Hit]:
        raise NotImplementedError
