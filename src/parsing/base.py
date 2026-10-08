"""解析器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Section


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> list[Section]:
        """把单个文件解析成带层级信息的 Section 列表。"""
        raise NotImplementedError
