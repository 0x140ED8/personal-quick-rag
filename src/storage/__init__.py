"""存储层。"""
from src.registry import registry
from src.storage.faiss_store import FAISSStore

registry.register("store", "faiss", lambda **kw: FAISSStore(**kw))
