"""DoclingParser：通用文档解析（基于 Docling 结构化输出）。

- 文本类（.txt/.md 等）：直接 UTF-8 读取，交给 HeadingExtractor 用正则重建层级。
- 其余格式（PDF/DOCX/PPTX/XLSX/EPUB…）：走 Docling，但**不使用扁平化 Markdown**，
  而是遍历结构化 items，从而：
  1. 用 SECTION_HEADER.level 还原真实标题层级（解决「所有标题都是二级」的问题）；
  2. 过滤 PAGE_HEADER / PAGE_FOOTER（解决页脚页码「104」混入正文的问题）；
  3. 表格单独导出，支持 TableFormer ACCURATE 模式改善合并单元格识别；
  4. 可选跳过目录（TOC）这类由点线引导符构成的大表格。
"""
from __future__ import annotations

import re
from pathlib import Path

from docling_core.types.doc.labels import DocItemLabel

from src.parsing.base import BaseParser
from src.parsing.heading_extractor import HeadingExtractor
from src.types import Section

# 直接读取文本的扩展名，避免无意义的版面解析
_TEXT_FORMATS = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".log"}

_HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
_CONTENT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
    DocItemLabel.CAPTION,
    DocItemLabel.REFERENCE,
    DocItemLabel.KEY_VALUE_REGION,
}


class DoclingParser(BaseParser):
    def __init__(
        self,
        heading_patterns: list = None,
        ocr: bool = False,
        table_mode: str = "accurate",
        skip_toc: bool = True,
        device: str = "cpu",
        markdown_dir: str = None,
    ) -> None:
        self._extractor = HeadingExtractor(heading_patterns)
        self._ocr = ocr
        self._table_mode = table_mode
        self._skip_toc = skip_toc
        self._device = device
        self._markdown_dir = markdown_dir
        self._converter = None

    def parse(self, path: str) -> list[Section]:
        p = Path(path)
        if p.suffix.lower() in _TEXT_FORMATS:
            text = p.read_text(encoding="utf-8", errors="ignore")
            return self._extractor.extract(text, source=p.name, page=None)
        return self._parse_structured(p)

    # ------------------------------------------------------------------ #
    def _parse_structured(self, p: Path) -> list[Section]:
        converter = self._get_converter()
        print(
            f"[1.解析] 开始转换 {p.name}（版面分析 + 表格识别 + 文本提取，可能需要数十秒）...",
            flush=True,
        )
        result = converter.convert(str(p))
        doc = result.document
        n_pages = len(doc.pages) if doc.pages else 0
        print(f"[1.解析] 转换完成，共 {n_pages} 页", flush=True)
        if self._markdown_dir:
            self._save_markdown(doc, p)
        return self._doc_to_sections(doc, p.name)

    def _doc_to_sections(self, doc, source: str) -> list[Section]:
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        buf: list[str] = []
        buf_path: list[str] = []
        buf_page = None
        stats = {"heading": 0, "table": 0, "text": 0, "skipped": 0}

        def flush() -> None:
            nonlocal buf, buf_path, buf_page
            if buf:
                sections.append(
                    Section(text="\n\n".join(buf), heading_path=list(buf_path), page=buf_page, source=source)
                )
            buf = []
            buf_path = []
            buf_page = None

        for item, _ in doc.iterate_items():
            label = item.label
            if label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                stats["skipped"] += 1
                continue

            page = self._page_of(item)

            if label in _HEADING_LABELS:
                stats["heading"] += 1
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                flush()
                if label == DocItemLabel.TITLE:
                    level = 0
                else:
                    level = self._heading_level(text, item, stack)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                buf = [text]
                buf_path = [t for _, t in stack]
                buf_page = page

            elif label == DocItemLabel.TABLE:
                stats["table"] += 1
                flush()
                md = self._table_markdown(item, doc)
                if not md:
                    continue
                if self._skip_toc and self._is_toc(md):
                    continue
                sections.append(
                    Section(text=md, heading_path=[t for _, t in stack], page=page, source=source)
                )

            elif label in _CONTENT_LABELS:
                stats["text"] += 1
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                if self._is_page_number(text):
                    continue
                if buf and buf_page is not None and page is not None and page != buf_page:
                    flush()
                if not buf:
                    buf_path = [t for _, t in stack]
                    buf_page = page
                buf.append(text)

        flush()
        print(
            f"[1.解析] 结构提取完成：{len(sections)} 段"
            f"（标题 {stats['heading']}、表格 {stats['table']}、"
            f"文本 {stats['text']}、跳过 {stats['skipped']}）",
            flush=True,
        )
        return sections

    def _save_markdown(self, doc, p: Path) -> None:
        """把 Docling 解析结果导出为 Markdown，便于人工检查解析质量。"""
        out_dir = Path(self._markdown_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (p.stem + ".md")
        try:
            md = doc.export_to_markdown()
        except Exception as e:  # noqa: BLE001
            print(f"[1.解析] 导出 Markdown 失败 {p.name}: {e}", flush=True)
            return
        out.write_text(md, encoding="utf-8")
        print(f"[1.解析] Markdown 已保存 -> {out}", flush=True)

    # ------------------------------------------------------------------ #
    def _get_converter(self):
        if self._converter is None:
            print(
                f"[1.解析] 初始化 Docling 转换器 (device={self._device}, table={self._table_mode}) ...",
                flush=True,
            )
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            mode = TableFormerMode.ACCURATE if self._table_mode == "accurate" else TableFormerMode.FAST
            pipeline_options = PdfPipelineOptions(
                do_ocr=self._ocr,
                do_table_structure=True,
                table_structure_options=TableStructureOptions(mode=mode, do_cell_matching=True),
                accelerator_options=AcceleratorOptions(device=self._device, num_threads=4),
            )
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            print("[1.解析] Docling 转换器就绪", flush=True)
        return self._converter

    def unload(self) -> None:
        """释放 Docling 转换器及其内部的版面/表格模型，回收显存/内存。"""
        if self._converter is not None:
            from src.utils import free_memory

            print("[1.解析] 释放 Docling 转换器以回收显存/内存 ...", flush=True)
            del self._converter
            self._converter = None
            free_memory()

    @staticmethod
    def _table_markdown(item, doc) -> str:
        try:
            return item.export_to_markdown(doc=doc) or ""
        except TypeError:
            return item.export_to_markdown() or ""

    @staticmethod
    def _page_of(item):
        prov = getattr(item, "prov", None)
        if prov:
            return int(prov[0].page_no)
        return None

    @staticmethod
    def _heading_level(text: str, item, stack: list[tuple[int, str]]) -> int:
        """推断标题层级。

        1. 编号型标题（如 2 / 2.1 / 2.3.1）按点分段数定级（最可靠）。
        2. 无编号标题：以 Docling 视觉层级为基准，但至少要比当前所在编号层级深一层，
           避免「Advanced-control timer」这类子标题被误判成顶级标题。
        """
        m = re.match(r"^(\d+(?:\.\d+)*)\b", text)
        if m:
            return m.group(1).count(".") + 1
        visual = int(getattr(item, "level", 1) or 1)
        numeric_depth = max((lvl for lvl, t in stack if re.match(r"^\d", t)), default=0)
        return max(visual, numeric_depth + 1)

    @staticmethod
    def _is_page_number(text: str) -> bool:
        """识别孤立页码（如页脚「104」「104/106」），这类文本是噪声。"""
        t = text.strip()
        if re.fullmatch(r"\d{1,4}", t):
            return True
        if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", t):
            return True
        return False

    @staticmethod
    def _is_toc(md: str) -> bool:
        """判断是否为目录表格：单元格含大量「点线引导符」（. . . .）且以页码结尾。"""
        cells = [c.strip() for c in md.split("|") if c.strip() and set(c.strip()) != {"-", ":"}]
        if not cells:
            return False
        dot_hits = sum(1 for c in cells if re.search(r"\.(\s*\.){3,}", c))
        return dot_hits >= 3
