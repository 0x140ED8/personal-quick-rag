"""跨模块通用数据结构（契约层）。

各阶段之间只依赖这里的类型，避免模块间直接耦合。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Section:
    """解析层产出：一段带层级归属与页码的文本。

    heading_path 的元素顺序按层级从大到小排列（如 ["第一卷", "第二章", "第三节"]）。
    """

    text: str
    heading_path: list = field(default_factory=list)
    page: Optional[int] = None
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "heading_path": list(self.heading_path),
            "page": self.page,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> "Section":
        return Section(
            text=d["text"],
            heading_path=list(d.get("heading_path", [])),
            page=d.get("page"),
            source=d.get("source", ""),
        )


@dataclass
class Chunk:
    """分片层产出：入库的最小文本单元。"""

    chunk_id: str
    text: str
    source: str = ""
    heading_path: list = field(default_factory=list)
    page: Optional[int] = None

    def to_metadata(self) -> dict:
        """转成向量库可序列化的标量元数据（FAISS 只接受标量）。"""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "heading_path": " > ".join(self.heading_path),
            "page": self.page if self.page is not None else -1,
        }

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "heading_path": list(self.heading_path),
            "page": self.page,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(
            chunk_id=d["chunk_id"],
            text=d["text"],
            source=d.get("source", ""),
            heading_path=list(d.get("heading_path", [])),
            page=d.get("page"),
        )


@dataclass
class Hit:
    """召回/重排层产出：最终返回给用户的结果。"""

    chunk_id: str
    text: str
    score: float = 0.0
    source: str = ""
    heading_path: list = field(default_factory=list)
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def heading_path_from_meta(value: str) -> list:
    """把存储层里拼接后的层级字符串还原为列表。"""
    if not value:
        return []
    return [p for p in value.split(" > ") if p]
