"""分片器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Chunk, Section


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, sections: list[Section], source: str = "") -> list[Chunk]:
        """把 Section 列表切分为 Chunk 列表，保留 heading_path / page 元数据。"""
        raise NotImplementedError
