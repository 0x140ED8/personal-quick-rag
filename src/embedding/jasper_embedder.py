"""Jasper 向量化：infgrad/Jasper-Token-Compression-600M 的 SentenceTransformer 直接封装。

为什么不复用 HuggingFaceEmbeddings：Jasper 需要区分查询/文档两侧的编码参数
（查询用模型内置 prompt_name="query"，且 encode 支持 compression_ratio），
LangChain 的 encode_kwargs 对 query/document 不区分，因此直接封装 SentenceTransformer。

模型采用**懒加载**：首次真正编码时才加载；unload() 释放显存/内存。

显存占用（8GB 笔记本显卡）：
- bfloat16 加载约 1.2GB，剩余约 6GB 可用于编码激活值；
- batch_size 建议 64 起（OOM 则调小）。
"""
from __future__ import annotations

import torch

from src.embedding.base import BaseEmbedder
from src.utils import cuda_allocated_mb, free_memory

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class _LangChainAdapter:
    """把 JasperEmbedder 适配成 LangChain Embeddings 接口，供 FAISS 存储层使用。"""

    def __init__(self, owner: "JasperEmbedder") -> None:
        self._owner = owner

    def embed_documents(self, texts: list) -> list:
        return self._owner.embed_documents(texts)

    def embed_query(self, text: str) -> list:
        return self._owner.embed_query(text)


class JasperEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "infgrad/Jasper-Token-Compression-600M",
        device: str = "auto",
        dtype: str = "bfloat16",
        batch_size: int = 64,
        compression_ratio: float = 0.3333,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._batch_size = batch_size
        self._compression_ratio = compression_ratio
        self._resolved_device = None
        self._model = None
        self._adapter = _LangChainAdapter(self)

    # ------------------------------------------------------------------ #
    # 懒加载
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        torch_dtype = _DTYPES.get(self._dtype)
        if torch_dtype is None:
            raise ValueError(f"未知 dtype: {self._dtype}，可选: {sorted(_DTYPES)}")

        print(
            f"[3.向量化] 加载模型 {self._model_name} "
            f"(device={device}, dtype={self._dtype}, "
            f"compression_ratio={self._compression_ratio}) ...",
            flush=True,
        )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            model_kwargs={
                "torch_dtype": torch_dtype,
                "attn_implementation": "sdpa",
                "trust_remote_code": True,
            },
            trust_remote_code=True,
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )
        print("[3.向量化] 模型加载完成", flush=True)
        if device == "cuda":
            used = cuda_allocated_mb()
            total = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
            print(f"[3.向量化] 当前已占用显存约 {used} MB / {total} MB", flush=True)

    def unload(self) -> None:
        """释放底层模型，把显存/内存还给系统（再次使用时自动重载）。"""
        if self._model is not None:
            print("[3.向量化] 释放模型以回收显存/内存 ...", flush=True)
            del self._model
            self._model = None
            free_memory()

    # ------------------------------------------------------------------ #
    # 接口
    # ------------------------------------------------------------------ #
    @property
    def raw(self):
        """LangChain Embeddings 适配器，供 FAISS 存储层序列化/反序列化使用（自动懒加载）。"""
        self._ensure_loaded()
        return self._adapter

    def embed_documents(self, texts: list) -> list:
        self._ensure_loaded()
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            compression_ratio=self._compression_ratio,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> list:
        self._ensure_loaded()
        vecs = self._model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
            compression_ratio=self._compression_ratio,
            show_progress_bar=False,
        )
        return vecs[0].tolist()
