"""生成层（占位）。"""
from src.registry import registry
from src.generation.none_generator import NoneGenerator

registry.register("generator", "none", lambda **kw: NoneGenerator(**kw))
