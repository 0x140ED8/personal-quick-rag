"""API 请求/响应模型（pydantic v2）。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# 通用
# ------------------------------------------------------------------ #
class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# ------------------------------------------------------------------ #
# /api/retrieve
# ------------------------------------------------------------------ #
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="最终返回片段数，默认取 reranker.top_k")
    rerank: bool = Field(True, description="是否用 BGE 重排（false 则直接返回混合召回结果）")


class HitItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    heading_path: List[str]
    page: Optional[int] = None


class RetrieveResponse(BaseModel):
    query: str
    hits: List[HitItem]
    took_ms: int


# ------------------------------------------------------------------ #
# /api/health /api/stats
# ------------------------------------------------------------------ #
class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    num_chunks: int
    embedder_loaded: bool
    reranker_loaded: bool
    ingest_running: bool
    uptime_sec: int


class DocStat(BaseModel):
    file: str
    size_bytes: int


class StatsResponse(BaseModel):
    num_chunks: int
    documents: List[DocStat]
    sources: List[str]
    components: dict
    build_meta: Optional[dict] = None


# ------------------------------------------------------------------ #
# /api/config
# ------------------------------------------------------------------ #
class ConfigUpdateResponse(BaseModel):
    message: str
    applied: bool
    warnings: List[str]
    backup: str


# ------------------------------------------------------------------ #
# /api/ingest / /api/refresh / /api/jobs/{id}
# ------------------------------------------------------------------ #
class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: str
    detail: str


class JobResponse(BaseModel):
    job_id: str
    status: str  # queued / running / done / failed
    stage: str
    done: int
    total: int
    detail: str
    created_at: float
    updated_at: float
    result: Optional[dict] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list, description="日志尾部（最近 50 行）")
