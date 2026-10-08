"""生成抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Hit


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, hits: list[Hit]) -> str:
        raise NotImplementedError
