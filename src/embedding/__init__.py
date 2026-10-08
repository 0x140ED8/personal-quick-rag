"""向量化层。"""
from src.registry import registry
from src.embedding.jasper_embedder import JasperEmbedder

registry.register("embedder", "jasper", lambda **kw: JasperEmbedder(**kw))
