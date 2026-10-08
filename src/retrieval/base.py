"""召回器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Hit


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        raise NotImplementedError
