"""分片层：把 Section 切成 Chunk，透传层级元数据。"""
from src.registry import registry
from src.chunking.recursive_chunker import RecursiveChunker

registry.register("chunker", "recursive", lambda **kw: RecursiveChunker(**kw))
