"""FAISS 向量存储：LangChain FAISS 封装。

- 入库用 MAX_INNER_PRODUCT（嵌入已归一化，等价于余弦相似度，分数越大越相关）。
- 查询走 search_by_vector，由上层负责把 query 编码成向量。
- save/load 使用 save_local / load_local 持久化到磁盘。
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.storage.base import BaseVectorStore
from src.types import Chunk, Hit, heading_path_from_meta


class _StoredVectorsEmbeddings(Embeddings):
    """占位 embedding：from_vectors 直接接收预算好的向量，不会调用它。

    查询路径是 search_by_vector；store.load() 时会被真实 embedder 替换，
    因此纯复用缓存场景无需加载 embedding 模型。
    """

    def embed_documents(self, texts: list) -> list:
        raise RuntimeError("该 FAISS 实例由 from_vectors 构建，未绑定 embedding 模型")

    def embed_query(self, text: str) -> list:
        raise RuntimeError("该 FAISS 实例由 from_vectors 构建，未绑定 embedding 模型")


class FAISSStore(BaseVectorStore):
    def __init__(self, distance_strategy: str = "MAX_INNER_PRODUCT") -> None:
        self._index = None
        self._distance_strategy = distance_strategy

    def add_documents(self, chunks: list[Chunk], embedder, batch_size: int = 16, progress=None) -> None:
        """分批编码并入库，每批完成后回调 progress(done, total) 用于进度展示。"""
        docs = [Document(page_content=c.text, metadata=c.to_metadata()) for c in chunks]
        emb = embedder.raw
        batch_size = max(1, int(batch_size))
        total = len(docs)
        text_embeddings = []
        for i in range(0, total, batch_size):
            batch = docs[i : i + batch_size]
            vecs = emb.embed_documents([d.page_content for d in batch])
            text_embeddings.extend(zip((d.page_content for d in batch), vecs))
            if progress is not None:
                progress(min(i + len(batch), total), total)
        if self._index is None:
            self._index = FAISS.from_embeddings(
                text_embeddings,
                emb,
                metadatas=[d.metadata for d in docs],
                distance_strategy=self._distance_strategy,
            )
        else:
            self._index.add_embeddings(text_embeddings, metadatas=[d.metadata for d in docs])

    def from_vectors(self, vectors, chunks: list[Chunk]) -> None:
        """用预算好的向量整体重建索引（不编码、不加载 embedding 模型）。"""
        if not chunks:
            raise ValueError("chunks 为空，无法建库")
        docs = [Document(page_content=c.text, metadata=c.to_metadata()) for c in chunks]
        text_embeddings = list(zip((c.text for c in chunks), vectors))
        self._index = FAISS.from_embeddings(
            text_embeddings,
            _StoredVectorsEmbeddings(),
            metadatas=[d.metadata for d in docs],
            distance_strategy=self._distance_strategy,
        )

    def search_by_vector(self, vector: list, top_k: int) -> list[Hit]:
        if self._index is None:
            return []
        results = self._index.similarity_search_with_score_by_vector(vector, k=top_k)
        return [self._to_hit(doc, score) for doc, score in results]

    def save(self, path) -> None:
        self._index.save_local(str(path))

    def load(self, path, embedder) -> None:
        self._index = FAISS.load_local(
            str(path), embedder.raw, allow_dangerous_deserialization=True
        )

    @staticmethod
    def _to_hit(doc: Document, score: float) -> Hit:
        m = doc.metadata or {}
        page = m.get("page")
        return Hit(
            chunk_id=m.get("chunk_id", ""),
            text=doc.page_content,
            score=float(score),
            source=m.get("source", ""),
            heading_path=heading_path_from_meta(m.get("heading_path", "")),
            page=None if page is None or page == -1 else int(page),
        )
