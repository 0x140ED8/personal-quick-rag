"""占位生成器：不接真实 LLM，仅把重排后的结果格式化输出。"""
from __future__ import annotations

from src.generation.base import BaseGenerator
from src.types import Hit


class NoneGenerator(BaseGenerator):
    def generate(self, query: str, hits: list[Hit]) -> str:
        if not hits:
            return "（生成器未接入）未检索到相关内容。"
        lines = ["（生成器未接入，以下为重排后的检索结果）"]
        for i, h in enumerate(hits, 1):
            path = " > ".join(h.heading_path) if h.heading_path else "（无层级）"
            page = f" 第{h.page}页" if h.page is not None else ""
            lines.append(f"[{i}] {h.source}{page} | {path}\n    {h.text}")
        return "\n".join(lines)
