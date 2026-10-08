"""BGE Reranker 重排：BAAI/bge-reranker-v2-m3（Cross-encoder）。

复刻 bge-reranker-v2-m3.py 的批处理打分，返回按重排分数降序的 Hit。
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.reranking.base import BaseReranker
from src.types import Hit


class BgeReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        print(f"[6.重排] 加载模型 {self._model_name} (device={self._device}) ...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval().to(self._device)
        print("[6.重排] 模型加载完成", flush=True)

    def unload(self) -> None:
        """释放重排模型占用的显存/内存（再次使用时自动重载）。"""
        if self._model is not None or self._tokenizer is not None:
            from src.utils import free_memory

            print("[6.重排] 释放模型以回收显存/内存 ...", flush=True)
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            free_memory()

    def rerank(self, query: str, hits: list[Hit], top_k: int = None) -> list[Hit]:
        if not hits:
            return []
        self._ensure_loaded()
        pairs = [[query, h.text] for h in hits]
        scores = self._score_pairs(pairs)
        scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return [
            Hit(
                chunk_id=h.chunk_id,
                text=h.text,
                score=float(s),
                source=h.source,
                heading_path=h.heading_path,
                page=h.page,
            )
            for h, s in scored
        ]

    def _score_pairs(self, pairs: list) -> list:
        all_scores = []
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            with torch.no_grad():
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self._max_length,
                ).to(self._device)
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(logits.cpu().tolist())
        return all_scores
