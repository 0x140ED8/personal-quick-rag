"""递归切分器：基于 LangChain 的 RecursiveCharacterTextSplitter。

按分隔符层级（段落 → 换行 → 空格 → 字符）逐级回退切分，chunk_id 由内容哈希生成，
保证跨次运行稳定（供 RRF 去重使用）。
"""
from __future__ import annotations

import hashlib
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.base import BaseChunker
from src.types import Chunk, Section


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list] = None,
    ) -> None:
        kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        if separators:
            kwargs["separators"] = separators
        self._splitter = RecursiveCharacterTextSplitter(**kwargs)

    def chunk(self, sections: list[Section], source: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in sections:
            src = sec.source or source
            pieces = self._splitter.split_text(sec.text)
            for i, text in enumerate(pieces):
                cid = self._make_id(src, sec.page, sec.heading_path, text)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        text=text,
                        source=src,
                        heading_path=list(sec.heading_path),
                        page=sec.page,
                    )
                )
        return chunks

    @staticmethod
    def _make_id(source: str, page: Optional[int], heading_path: list, text: str) -> str:
        raw = "|".join([source, str(page if page is not None else -1), ">".join(heading_path), text])
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
