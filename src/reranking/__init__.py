"""重排序层。"""
from src.registry import registry
from src.reranking.bge_reranker import BgeReranker

registry.register("reranker", "bge", lambda **kw: BgeReranker(**kw))
