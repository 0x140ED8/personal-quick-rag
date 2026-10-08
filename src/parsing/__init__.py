"""解析层：把原始文档解析为带结构信息的 Section 列表。"""
from src.registry import registry
from src.parsing.docling_parser import DoclingParser

registry.register("parser", "docling", lambda **kw: DoclingParser(**kw))
