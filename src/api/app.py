"""FastAPI 应用工厂：检索 / 配置 / 索引任务 全套路由。

设计要点：
- 单进程单 worker + 全局锁：GPU 推理不宜并发，检索与 ingest 互斥；
  ingest 运行期间的检索请求直接返回 409，而不是排队挂起。
- 模型常驻：服务模式下不走 pipeline.query() 的 unload 逻辑，
  embedder/reranker 一次加载常驻显存（合计约 3.4GB，8GB 卡可容纳）。
- 配置懒重建：POST /api/config 写入文件后，模型类配置变更会重建
  pipeline（组件懒加载，重组装成本低），检索参数下一请求即生效。
"""
from __future__ import annotations

import copy
import io
import json
import threading
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.config import (
    Config,
    config_from_dict,
    config_to_dict,
    load_config,
    validate_config_dict,
)
from src.pipeline import RAGPipeline, IngestProgress
from src.registry import registry
from src.types import Hit

from src.api import schemas

# 任务与日志保留上限
_MAX_JOBS = 20
_MAX_LOG_LINES = 2000
_LOG_TAIL = 50
# 已知组件名（供配置校验给出可操作提示）
_KNOWN_COMPONENTS = {
    "parser": {"docling"},
    "chunker": {"recursive"},
    "embedder": {"jasper"},
    "store": {"faiss"},
    "retriever": {"hybrid"},
    "reranker": {"bge"},
    "generator": {"none"},
}
# 影响模型/组件装配的配置段（变更需重建 pipeline）
_PIPELINE_SECTIONS = {"paths", "parser", "chunker", "embedder", "store", "retriever", "reranker", "generator"}


def _api_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


class UTF8JSONResponse(JSONResponse):
    """显式声明 charset=utf-8：PowerShell 5.1 的 Invoke-RestMethod 在无 charset 时
    会按 Latin-1 解码 UTF-8 JSON，导致中文乱码（如 config 回读回写场景）。"""

    media_type = "application/json; charset=utf-8"


class IngestJob:
    """一次后台 ingest 任务：状态 + 阶段进度 + 捕获的日志。"""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status = "queued"  # queued / running / done / failed
        self.progress = IngestProgress()
        self.created_at = time.time()
        self.updated_at = time.time()
        self.result: dict | None = None
        self.error: str | None = None
        self.logs: list = []

    def log_line(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > _MAX_LOG_LINES:
            del self.logs[: len(self.logs) - _MAX_LOG_LINES]
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.progress.stage,
            "done": self.progress.done,
            "total": self.progress.total,
            "detail": self.progress.detail,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "logs": self.logs[-_LOG_TAIL:],
        }


class _LogTap(io.TextIOBase):
    """把 print 输出逐行喂给 IngestJob（同时透传到原 stdout）。"""

    def __init__(self, job: IngestJob, fallback) -> None:
        self._job = job
        self._fallback = fallback

    def write(self, s: str) -> int:
        try:
            self._fallback.write(s)
        except Exception:  # noqa: BLE001
            pass
        for line in s.splitlines():
            if line.strip():
                self._job.log_line(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._fallback.flush()
        except Exception:  # noqa: BLE001
            pass


class AppState:
    """共享状态：配置、pipeline、锁、任务表。"""

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(str(self.config_path))
        self.pipeline = RAGPipeline(self.config, registry)
        # 注意：必须用 Lock（非 RLock）——ingest 请求线程获取、后台线程释放，
        # RLock 不允许非持有线程释放
        self.lock = threading.Lock()
        self.jobs: dict = {}
        self.jobs_order: list = []
        self.ingest_job_id: str | None = None  # 当前运行中的 ingest 任务
        self.started_at = time.time()

    # ------------------------------ 配置 ------------------------------ #
    def read_config_file(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def write_config_file(self, data: dict) -> None:
        header = (
            "# 本文件由 POST /api/config 自动生成，原有注释已丢失。\n"
            "# 生成时间: "
            + datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            + "\n"
        )
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def rebuild_pipeline(self) -> None:
        """按当前 self.config 重建组件（先卸载旧组件释放显存）。"""
        old = self.pipeline
        old._parser.unload()
        old._embedder.unload()
        old._reranker.unload()
        self.pipeline = RAGPipeline(self.config, registry)

    # ------------------------------ 任务 ------------------------------ #
    def register_job(self, job: IngestJob) -> None:
        self.jobs[job.job_id] = job
        self.jobs_order.append(job.job_id)
        while len(self.jobs_order) > _MAX_JOBS:
            old_id = self.jobs_order.pop(0)
            self.jobs.pop(old_id, None)


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def create_app(config_path: str = "config.yaml") -> FastAPI:
    # 触发各子包注册组件（与 main.py 保持一致）
    import src.parsing  # noqa: F401
    import src.chunking  # noqa: F401
    import src.embedding  # noqa: F401
    import src.storage  # noqa: F401
    import src.reranking  # noqa: F401
    import src.generation  # noqa: F401

    state = AppState(config_path)

    app = FastAPI(
        title="RAG 知识库检索服务",
        description="纯检索服务：文档解析/混合召回/重排；LLM 生成由调用方负责。",
        version="1.0.0",
        default_response_class=UTF8JSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=state.config.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 统一错误格式 {"error": {"code", "message"}}
    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):  # noqa: ANN001
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError):  # noqa: ANN001
        return _api_error(400, "INVALID_BODY", f"请求体校验失败: {exc.errors()[:3]}")

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(request: Request, exc: Exception):  # noqa: ANN001
        return _api_error(500, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # 健康检查
    # ------------------------------------------------------------------ #
    @app.get("/api/health", response_model=schemas.HealthResponse, tags=["system"])
    def health() -> dict:
        loaded = state.pipeline.components_loaded()
        return {
            "status": "ok" if state.pipeline.index_ready() else "degraded",
            "index_ready": state.pipeline.index_ready(),
            "num_chunks": state.pipeline.num_chunks(),
            "embedder_loaded": loaded["embedder"],
            "reranker_loaded": loaded["reranker"],
            "ingest_running": state.ingest_job_id is not None,
            "uptime_sec": int(time.time() - state.started_at),
        }

    # ------------------------------------------------------------------ #
    # 知识库统计
    # ------------------------------------------------------------------ #
    @app.get("/api/stats", response_model=schemas.StatsResponse, tags=["system"])
    def stats() -> dict:
        cfg = state.config
        docs = [
            {"file": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(Path(cfg.paths.data_dir).iterdir())
            if p.is_file()
        ]
        sources = []
        chunks_file = Path(cfg.paths.index_dir) / "chunks.jsonl"
        if chunks_file.exists():
            seen = set()
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            s = json.loads(line).get("source", "")
                        except Exception:  # noqa: BLE001
                            continue
                        if s and s not in seen:
                            seen.add(s)
                            sources.append(s)
        components = {
            "parser": cfg.parser.name,
            "chunker": cfg.chunker.name,
            "embedder": f"{cfg.embedder.name} ({cfg.embedder.model_name})",
            "store": cfg.store.name,
            "retriever": cfg.retriever.name,
            "reranker": f"{cfg.reranker.name} ({cfg.reranker.model_name})",
            "generator": cfg.generator.name,
        }
        return {
            "num_chunks": state.pipeline.num_chunks(),
            "documents": docs,
            "sources": sources,
            "components": components,
            "build_meta": state.pipeline.load_build_meta(),
        }

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    @app.post("/api/retrieve", response_model=schemas.RetrieveResponse, tags=["retrieval"])
    def retrieve(body: schemas.RetrieveRequest) -> dict:
        if state.ingest_job_id is not None:
            raise HTTPException(409, "索引构建进行中，请稍后通过 /api/jobs 查询进度后再试")
        if not state.pipeline.index_ready():
            raise HTTPException(404, "索引不存在，请先 POST /api/ingest 建立索引")
        acquired = state.lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(409, "服务正忙（另一请求处理中），请稍后重试")
        try:
            t0 = time.time()
            hits: list[Hit] = state.pipeline.retrieve(body.query, top_k=body.top_k, rerank=body.rerank)
            took_ms = int((time.time() - t0) * 1000)
            return {
                "query": body.query,
                "hits": [h.to_dict() for h in hits],
                "took_ms": took_ms,
            }
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # 配置查询 / 修改
    # ------------------------------------------------------------------ #
    @app.get("/api/config", tags=["config"])
    def get_config() -> dict:
        # 直接读文件转 JSON（与磁盘上的 config.yaml 一致，不含注释）
        return state.read_config_file()

    @app.post("/api/config", response_model=schemas.ConfigUpdateResponse, tags=["config"])
    def update_config(body: dict) -> dict:
        if state.ingest_job_id is not None:
            raise HTTPException(409, "索引构建进行中，禁止修改配置")
        acquired = state.lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(409, "服务正忙（另一请求处理中），请稍后重试")
        try:
            return _apply_config(state, body)
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # 索引构建 / 增量刷新（后台任务，共用同一任务槽与处理逻辑）
    # ------------------------------------------------------------------ #
    @app.post("/api/refresh", response_model=schemas.IngestAcceptedResponse, status_code=202, tags=["ingest"])
    def refresh() -> dict:
        """执行一遍增量更新（等价于 /api/ingest：只处理新增/变更文件，其余复用缓存）。"""
        return _start_ingest_job(state, "增量刷新")

    @app.post("/api/ingest", response_model=schemas.IngestAcceptedResponse, status_code=202, tags=["ingest"])
    def start_ingest() -> dict:
        """触发重建索引（增量，后台任务）。"""
        return _start_ingest_job(state, "索引构建")

    @app.get("/api/jobs/{job_id}", response_model=schemas.JobResponse, tags=["ingest"])
    def get_job(job_id: str) -> dict:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"任务不存在: {job_id}")
        return job.to_dict()

    @app.get("/api/jobs", tags=["ingest"])
    def list_jobs() -> dict:
        return {
            "jobs": [
                {"job_id": jid, "status": state.jobs[jid].status, "created_at": _utc_iso(state.jobs[jid].created_at)}
                for jid in reversed(state.jobs_order)
            ]
        }

    # ------------------------------------------------------------------ #
    # 启动预热
    # ------------------------------------------------------------------ #
    @app.on_event("startup")
    def warmup() -> None:
        if not state.config.server.warmup:
            print("[服务] warmup=false，跳过预加载", flush=True)
            return
        if not state.pipeline.index_ready():
            print("[服务] 索引不存在，跳过预加载（请先 POST /api/ingest）", flush=True)
            return
        try:
            print("[服务] 预加载索引与模型 ...", flush=True)
            state.pipeline.warmup()
            print("[服务] 预加载完成", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[服务] 预加载失败（不影响启动）: {e}", flush=True)

    return app


def sys_stdout():
    import sys

    return sys.__stdout__ or sys.stdout


def _start_ingest_job(state: AppState, action: str) -> dict:
    """启动后台增量 ingest 任务（/api/ingest 与 /api/refresh 共用）。

    同一时间只允许一个构建任务（state.ingest_job_id 作为任务槽），
    后台线程结束时释放锁；锁被检索/改配置占用则直接 409。
    """
    if state.ingest_job_id is not None:
        raise HTTPException(409, f"已有构建任务运行中: {state.ingest_job_id}")
    acquired = state.lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(409, "服务正忙，请稍后重试")
    job = IngestJob(uuid.uuid4().hex[:12])
    state.register_job(job)
    state.ingest_job_id = job.job_id
    job.status = "running"

    def _run() -> None:
        tap = _LogTap(job, sys_stdout())
        try:
            with redirect_stdout(tap), redirect_stderr(tap):
                result = state.pipeline.ingest(progress=job.progress)
                job.result = result
                job.status = "done"
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.log_line(f"[任务失败] {job.error}")
        finally:
            state.ingest_job_id = None
            state.lock.release()

    threading.Thread(target=_run, name="ingest", daemon=True).start()
    return {
        "job_id": job.job_id,
        "status": "running",
        "detail": f"{action}已开始，通过 GET /api/jobs/{job.job_id} 查询进度",
    }


def _apply_config(state: AppState, body: dict) -> dict:
    """校验 -> 备份 -> 写回 -> 懒重建。"""
    # 1. 结构校验（未知段落/字段/类型）
    try:
        normalized = validate_config_dict(body)
    except ValueError as e:
        raise HTTPException(400, f"配置校验失败: {e}") from e

    # 2. 组件名合法性（给更可操作的提示）
    for section, names in _KNOWN_COMPONENTS.items():
        if section in normalized and "name" in normalized[section]:
            n = normalized[section]["name"]
            if n not in names:
                raise HTTPException(400, f"段落 {section} 的 name={n!r} 未知，可选: {sorted(names)}")

    # 3. 与当前配置对比，决定生效方式
    current = config_to_dict(state.config)
    warnings = []
    changed_sections = [
        s for s in normalized if normalized[s] != current.get(s)
    ] or [s for s in current if s not in normalized]
    if not changed_sections:
        return {"message": "配置无变化", "applied": True, "warnings": [], "backup": ""}

    pipeline_changed = [s for s in changed_sections if s in _PIPELINE_SECTIONS]

    # 4. 备份 + 写回（safe_dump 不保留注释）
    backup = ""
    if state.config_path.exists():
        backup = str(state.config_path) + ".bak"
        copy_config_file(str(state.config_path), backup)
    state.write_config_file(normalized)

    # 5. 生效
    new_config = config_from_dict(normalized)
    old_embedder = (state.config.embedder.name, state.config.embedder.model_name)
    new_embedder = (new_config.embedder.name, new_config.embedder.model_name)

    state.config = new_config
    if pipeline_changed:
        state.rebuild_pipeline()
        warnings.append(f"已重建组件（配置段变更: {pipeline_changed}），模型将在下次使用时懒加载")
    if old_embedder != new_embedder:
        warnings.append(
            f"embedder 已从 {old_embedder[0]}/{old_embedder[1]} 改为 "
            f"{new_embedder[0]}/{new_embedder[1]}，现有索引向量空间不匹配，"
            "检索将报错，需 POST /api/ingest 重建索引"
        )
    if "paths" in changed_sections:
        warnings.append("paths 配置已变更，索引目录指向新位置，请确认索引存在或重新 ingest")
    if "server" in changed_sections:
        old_s, new_s = current.get("server", {}), normalized["server"]
        if old_s.get("host") != new_s.get("host") or old_s.get("port") != new_s.get("port"):
            warnings.append("server.host/port 变更需重启服务后生效")
        if old_s.get("warmup") != new_s.get("warmup"):
            warnings.append("server.warmup 变更需重启服务后生效")

    return {
        "message": "配置已写入并生效",
        "applied": True,
        "warnings": warnings,
        "backup": backup,
    }


def copy_config_file(src: str, dst: str) -> None:
    Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
