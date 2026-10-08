"""结构提取器：从 Markdown 文本中重建标题层级树，为每段打 heading_path。

支持两类标题：
1. Markdown 标题：`#{1,6} 标题`（Docling 解析 PDF/DOCX 等结构化文档时产出）。
2. 中文卷章节模式：可配置正则（如小说《斗破苍穹》的「第一章 xxx」）。

heading_path 的元素顺序按层级从大到小排列（如 ["第一章", "第一节"]）。
"""
from __future__ import annotations

import re
from typing import Optional

from src.types import Section


class HeadingExtractor:
    def __init__(self, heading_patterns: Optional[list] = None) -> None:
        self._patterns: list[tuple] = []
        for p in heading_patterns or []:
            if isinstance(p, dict):
                regex = p.get("regex")
                level = int(p.get("level", 1))
            else:
                regex = str(p)
                level = 1
            if regex:
                self._patterns.append((re.compile(regex), level))
        self._patterns.sort(key=lambda x: x[1])

    def extract(self, markdown: str, source: str = "", page: Optional[int] = None) -> list[Section]:
        return self.extract_segments([(markdown, page)], source=source)

    def extract_segments(self, segments: list, source: str = "") -> list[Section]:
        """依次处理多个 (markdown, page) 片段，标题栈跨片段保留。

        这样跨页的章节不会在页边界丢失层级归属，同时每段能带上准确页码。
        """
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        buf: list[str] = []
        buf_path: list[str] = []
        buf_page: Optional[int] = None

        def flush() -> None:
            nonlocal buf, buf_path, buf_page
            if buf:
                sections.append(
                    Section(text="\n".join(buf), heading_path=list(buf_path), page=buf_page, source=source)
                )
            buf = []
            buf_path = []
            buf_page = None

        for md, page in segments:
            if not md:
                continue
            for raw in md.splitlines():
                line = raw.rstrip()
                heading = self._match_heading(line)
                if heading is not None:
                    flush()
                    level, title = heading
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    buf = [line.strip()]
                    buf_path = [t for _, t in stack]
                    buf_page = page
                else:
                    if line.strip():
                        if not buf:
                            buf_page = page
                        buf.append(line)
            flush()

        return sections

    def _match_heading(self, line: str):
        stripped = line.strip()
        if not stripped:
            return None
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            return len(m.group(1)), m.group(2).strip()
        for pattern, level in self._patterns:
            if pattern.match(stripped):
                return level, stripped
        return None
