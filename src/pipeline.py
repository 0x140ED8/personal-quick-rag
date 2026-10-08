"""管线组装：把各阶段组件按配置装配，提供 ingest / retrieve / query 入口。

ingest 采用「阶段制」批处理 + 按文件增量缓存：未变更文件直接复用
index_dir/cache/ 下的解析/分片/向量缓存，只对新增或变更文件走
解析 → 切片 → 向量化；FAISS 仍用「缓存向量 + 新向量」整体重建，
失败时旧索引不被破坏。进度通过 progress 回调上报。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.cache import FileCacheManager
from src.config import Config
from src.retrieval import HybridRetriever, VectorRetriever, BM25Retriever
from src.types import Chunk, Hit
from src.utils import ProgressBar

# ingest 各阶段的标识（任务 API 的 stage 字段取值）
STAGE_PARSING = "parsing"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_SAVING = "saving"
STAGE_DONE = "done"


class IngestProgress:
    """进度回调对象：记录当前阶段与阶段内进度，供任务 API 查询。"""

    def __init__(self) -> None:
        self.stage = STAGE_PARSING
        self.done = 0
        self.total = 0
        self.detail = ""

    def update(self, stage: str, done: int, total: int, detail: str = "") -> None:
        self.stage = stage
        self.done = int(done)
        self.total = int(total)
        self.detail = detail

    def to_dict(self) -> dict:
        return {"stage": self.stage, "done": self.done, "total": self.total, "detail": self.detail}


class RAGPipeline:
    def __init__(self, config: Config, registry) -> None:
        print("[初始化] 组装各阶段组件 ...", flush=True)
        self._config = config
        self._registry = registry
        self._index_dir = Path(config.paths.index_dir)
        self._data_dir = Path(config.paths.data_dir)
        self._chunks_file = self._index_dir / "chunks.jsonl"
        self._faiss_dir = self._index_dir / "faiss"
        self._meta_file = self._index_dir / "build_meta.json"
        self._build_components()
        self._loaded = False
        print("[初始化] 组件组装完成", flush=True)

    def _build_components(self) -> None:
        """按当前配置实例化各阶段组件（模型懒加载，重组装成本很低）。"""
        config = self._config
        registry = self._registry

        # 1. 解析
        self._parser = registry.create(
            "parser",
            config.parser.name,
            heading_patterns=config.parser.heading_patterns,
            ocr=config.parser.ocr,
            table_mode=config.parser.table_mode,
            skip_toc=config.parser.skip_toc,
            device=config.parser.device,
            markdown_dir=config.paths.parsed_dir,
        )
        # 2. 分片
        self._chunker = registry.create(
            "chunker",
            config.chunker.name,
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
        )
        # 3. 向量化
        self._embedder = registry.create(
            "embedder",
            config.embedder.name,
            model_name=config.embedder.model_name,
            device=config.embedder.device,
            dtype=config.embedder.dtype,
            batch_size=config.embedder.batch_size,
            compression_ratio=config.embedder.compression_ratio,
        )
        # 4. 存储
        self._store = registry.create(
            "store", config.store.name, distance_strategy=config.store.distance_strategy
        )

        # 5. 召回（稠密 + 稀疏 + 融合）
        self._bm25 = BM25Retriever(chunks_file=str(self._chunks_file))
        dense = VectorRetriever(self._store, self._embedder)
        self._retriever = HybridRetriever(
            dense,
            self._bm25,
            dense_top_k=config.retriever.dense_top_k,
            sparse_top_k=config.retriever.sparse_top_k,
            rrf_k=config.retriever.rrf_k,
        )

        # 6. 重排
        self._reranker = registry.create(
            "reranker",
            config.reranker.name,
            model_name=config.reranker.model_name,
            device=config.reranker.device,
        )

        # 7. 生成
        self._generator = registry.create("generator", config.generator.name)

    # ------------------------------------------------------------------ #
    # 离线：建立索引（增量：未变文件复用缓存，新增/变更文件走完整流程）
    # ------------------------------------------------------------------ #
    def ingest(self, progress: IngestProgress = None) -> dict:
        if progress is None:
            progress = IngestProgress()
        result = {
            "chunks": 0,
            "files": 0,
            "failed_files": [],
            "took_sec": 0.0,
            "files_reused": 0,
            "files_added": 0,
            "files_removed": 0,
            "chunks_reused": 0,
            "chunks_embedded": 0,
        }
        t0 = time.time()
        print("=" * 60)
        print("开始建立索引 (ingest，未变文件复用缓存)")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        # 重建存储组件：serve 模式下 store 可能已加载旧索引（warmup/上次检索），
        # 必须换新实例，否则新 chunk 会追加到旧索引而非整体重建
        self._store = self._registry.create(
            "store", self._config.store.name, distance_strategy=self._config.store.distance_strategy
        )
        self._loaded = False
        files = sorted(p for p in self._data_dir.iterdir() if p.is_file())

        # ---- 阶段 0：扫描缓存，分类 复用 / 新增(变更) / 删除 ----
        cache = FileCacheManager(self._index_dir / "cache")
        cache.load()
        fingerprints = FileCacheManager.fingerprints(
            self._config.parser, self._config.chunker, self._config.embedder
        )
        print(
            f"[0.缓存] 缓存记录 {len(cache.manifest)} 条，数据文件 {len(files)} 个",
            flush=True,
        )
        data_names = {f.name for f in files}
        delete_names = [name for name in cache.manifest if name not in data_names]
        reused: dict[str, tuple[list, np.ndarray]] = {}
        add_files = []
        for f in files:
            key = cache.key_for(f, fingerprints)
            loaded = cache.read(f.name) if cache.manifest.get(f.name) == key else None
            if loaded is not None:
                _sections, chunks, vectors = loaded
                reused[f.name] = (chunks, vectors)
                print(f"[0.缓存] 复用 {f.name}: {len(chunks)} 块", flush=True)
            else:
                if f.name in cache.manifest:
                    delete_names.append(f.name)
                    print(f"[0.缓存] 失效 {f.name}（文件或参数已变更）", flush=True)
                add_files.append(f)
        # 先执行删除队列，再执行新增队列
        for name in sorted(set(delete_names)):
            cache.delete(name)
        result["files_reused"] = len(reused)
        result["files_removed"] = sum(1 for n in delete_names if n not in data_names)
        result["chunks_reused"] = sum(len(c) for c, _ in reused.values())

        # ---- 阶段 1：只解析新增/变更文件（Docling 只加载一次） ----
        parsed: dict = {}
        if add_files:
            print(
                f"[1.解析] 需解析 {len(add_files)} 个文件（其余 {len(reused)} 个复用缓存）",
                flush=True,
            )
            for i, f in enumerate(add_files, 1):
                progress.update(STAGE_PARSING, i, len(add_files), f.name)
                print(f"[1.解析] ({i}/{len(add_files)}) 开始处理 {f.name} ...", flush=True)
                try:
                    sections = self._parser.parse(str(f))
                except Exception as e:  # noqa: BLE001
                    print(f"[1.解析] 跳过 {f.name}: {e}")
                    result["failed_files"].append({"file": f.name, "error": str(e)})
                    continue
                parsed[f.name] = sections
                print(f"[1.解析] {f.name}: {len(sections)} 段", flush=True)
        else:
            print("[1.解析] 无新增/变更文件，跳过解析", flush=True)
            progress.update(STAGE_PARSING, 0, 0, "全部复用缓存")

        # 全部文件解析完成，统一释放 Docling 的版面/表格模型（embedding 模型尚未加载）
        self._parser.unload()

        if not files:
            print("未找到可解析的文档，请检查 data_dir。")
            result["took_sec"] = round(time.time() - t0, 1)
            progress.update(STAGE_DONE, 0, 0, "无可解析文档")
            return result
        result["files_added"] = len(parsed)

        # ---- 阶段 2：只对新增/变更文件切片（纯 CPU，无模型装卸） ----
        new_chunks: dict = {}
        total_new = 0
        if parsed:
            total_sections = sum(len(s) for s in parsed.values())
            progress.update(STAGE_CHUNKING, 0, total_sections, "")
            done_sections = 0
            for f in add_files:
                sections = parsed.get(f.name)
                if sections is None:  # 解析失败的文件
                    continue
                chunks = self._chunker.chunk(sections, source=f.name)
                new_chunks[f.name] = chunks
                total_new += len(chunks)
                done_sections += len(sections)
                progress.update(STAGE_CHUNKING, done_sections, total_sections, f.name)
                print(f"[2.分片] {f.name}: {len(sections)} 段 -> {len(chunks)} 块", flush=True)
            print(f"[2.分片] 完成，共 {total_new} 块")
        else:
            progress.update(STAGE_CHUNKING, 0, 0, "全部复用缓存")

        # ---- 阶段 3：只向量化新增 chunk，按文件写入缓存 ----
        new_vectors: dict[str, np.ndarray] = {}
        if parsed:
            progress.update(STAGE_EMBEDDING, 0, total_new, "")
            if total_new:
                print(
                    f"[3.向量化] 正在编码 {total_new} 个 chunk"
                    f"（复用 {result['chunks_reused']} 个） ...",
                    flush=True,
                )
            else:
                print("[3.向量化] 新增文件没有可编码的 chunk", flush=True)
            bar = ProgressBar(total=total_new, label="[3.向量化] 编码进度") if total_new else None
            emb = self._embedder.raw if total_new else None
            batch_size = max(1, int(self._config.embedder.batch_size))
            embedded_so_far = 0
            for f in add_files:
                chunks = new_chunks.get(f.name)
                if chunks is None:
                    continue
                vectors: list = []
                written = 0
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    vectors.extend(emb.embed_documents([c.text for c in batch]))
                    written += len(batch)
                    progress.update(STAGE_EMBEDDING, embedded_so_far + written, total_new, f.name)
                    bar.update(embedded_so_far + written, total_new)
                arr = np.asarray(vectors, dtype=np.float32)
                if arr.size == 0:
                    arr = arr.reshape(0, 0)
                embedded_so_far += len(chunks)
                new_vectors[f.name] = arr
                key = cache.key_for(self._data_dir / f.name, fingerprints)
                cache.write(f.name, key, parsed[f.name], chunks, arr)
                print(f"[3.向量化] {f.name}: {len(chunks)} 块已写入缓存", flush=True)
            if bar is not None:
                bar.close()
            self._embedder.unload()
            result["chunks_embedded"] = total_new
            print("[3.向量化] 完成", flush=True)
        else:
            print("[3.向量化] 全部复用缓存，跳过", flush=True)
            progress.update(STAGE_EMBEDDING, 0, 0, "全部复用缓存")

        # ---- 阶段 4：缓存向量 + 新向量拼接，整体重建索引 ----
        pairs: list = []
        for f in files:
            if f.name in new_chunks:
                pairs.append((new_chunks[f.name], new_vectors[f.name]))
            elif f.name in reused:
                pairs.append(reused[f.name])
        all_chunks: list[Chunk] = []
        vector_groups: list[np.ndarray] = []
        for chunks, vectors in pairs:
            all_chunks.extend(chunks)
            if len(chunks) > 0:
                vector_groups.append(vectors)
        if not all_chunks:
            print("未找到可解析的文档，请检查 data_dir。")
            result["took_sec"] = round(time.time() - t0, 1)
            progress.update(STAGE_DONE, 0, 0, "无可解析文档")
            return result
        all_vectors = np.vstack(vector_groups)
        print(
            f"[4.存储] 整体重建 FAISS 索引（共 {len(all_chunks)} 块，新编码 {total_new} 块）",
            flush=True,
        )
        self._store.from_vectors(all_vectors, all_chunks)
        progress.update(STAGE_SAVING, 0, 3, "FAISS")
        print("[4.存储] 保存 FAISS 索引 ...", flush=True)
        self._store.save(self._faiss_dir)
        progress.update(STAGE_SAVING, 1, 3, "BM25")
        print("[5.BM25] 构建稀疏索引 ...", flush=True)
        self._bm25.save_chunks(all_chunks, str(self._chunks_file))
        self._bm25.build(all_chunks)
        progress.update(STAGE_SAVING, 2, 3, "build_meta")
        self._save_build_meta(len(all_chunks))
        progress.update(STAGE_SAVING, 3, 3, "done")

        # ingest 重建了索引，但组件引用未变（懒加载），标记需重新加载索引数据
        self._loaded = False
        result["chunks"] = len(all_chunks)
        result["files"] = result["files_reused"] + result["files_added"]
        result["took_sec"] = round(time.time() - t0, 1)
        orphans = cache.prune_orphans()
        if orphans:
            print(f"[0.缓存] 清理孤儿缓存 {orphans} 个", flush=True)
        print(
            f"索引完成：共 {len(all_chunks)} 个 chunk"
            f"（复用 {result['chunks_reused']} / 新编码 {result['chunks_embedded']}），"
            f"文件 复用 {result['files_reused']} / 新增 {result['files_added']} /"
            f" 删除 {result['files_removed']}",
            flush=True,
        )
        print("=" * 60)
        return result

    # ------------------------------------------------------------------ #
    # 在线：召回 + 重排
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = None, rerank: bool = True) -> list[Hit]:
        self._ensure_loaded()
        self._ensure_index_compatible()
        print("[5.召回] 混合检索 (dense+BM25+RRF) ...", flush=True)
        candidates = self._retriever.retrieve(query, self._config.retriever.dense_top_k)
        print(f"[5.召回] 候选 {len(candidates)} 条")
        if not rerank:
            if top_k is None:
                top_k = self._config.reranker.top_k
            return candidates[:top_k]
        if top_k is None:
            top_k = self._config.reranker.top_k
        print(f"[6.重排] 对候选重排，取 top {top_k} ...", flush=True)
        return self._reranker.rerank(query, candidates, top_k)

    def query(self, query: str, top_k: int = None) -> str:
        try:
            hits = self.retrieve(query, top_k)
            return self._generator.generate(query, hits)
        finally:
            # CLI 查询流程结束，释放召回/重排模型占用的显存/内存
            # （API 服务模式不走这里：直接调 retrieve()，模型常驻）
            self._embedder.unload()
            self._reranker.unload()

    # ------------------------------------------------------------------ #
    # 索引元信息（embedder 兼容性校验）
    # ------------------------------------------------------------------ #
    def _save_build_meta(self, num_chunks: int) -> None:
        meta = {
            "embedder_name": self._config.embedder.name,
            "embedder_model_name": self._config.embedder.model_name,
            "num_chunks": num_chunks,
            "param_fingerprints": FileCacheManager.fingerprints(
                self._config.parser, self._config.chunker, self._config.embedder
            ),
        }
        self._meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[4.存储] 构建元信息已保存 -> {self._meta_file}", flush=True)

    def load_build_meta(self) -> dict | None:
        if not self._meta_file.exists():
            return None
        try:
            return json.loads(self._meta_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _ensure_index_compatible(self) -> None:
        """校验当前 embedder 配置与建库时一致，避免索引与查询向量空间不匹配。"""
        meta = self.load_build_meta()
        if meta is None:
            return  # 旧索引无元信息，跳过校验
        if (
            meta.get("embedder_name") != self._config.embedder.name
            or meta.get("embedder_model_name") != self._config.embedder.model_name
        ):
            raise RuntimeError(
                f"embedder 配置与索引不匹配：索引由 {meta.get('embedder_name')}/"
                f"{meta.get('embedder_model_name')} 构建，当前配置为 "
                f"{self._config.embedder.name}/{self._config.embedder.model_name}，"
                "请重新运行 ingest 或改回原配置"
            )
        meta_fp = meta.get("param_fingerprints")
        if isinstance(meta_fp, dict):
            current_fp = FileCacheManager.fingerprints(
                self._config.parser, self._config.chunker, self._config.embedder
            )
            changed = [k for k, v in current_fp.items() if meta_fp.get(k) != v]
            if changed:
                raise RuntimeError(
                    f"解析/分片/向量化参数已变更（{'/'.join(changed)}），索引与当前配置不匹配，"
                    "请重新运行 ingest（增量重建会自动复用未变文件的缓存）"
                )

    def index_ready(self) -> bool:
        return self._faiss_dir.exists() and self._chunks_file.exists()

    def warmup(self) -> None:
        """预加载索引与模型（API 服务启动时调用，避免首次检索延迟）。"""
        self._ensure_loaded()
        self._ensure_index_compatible()
        self._embedder.embed_query("warmup")
        ensure = getattr(self._reranker, "_ensure_loaded", None)
        if callable(ensure):
            ensure()

    def components_loaded(self) -> dict:
        """各模型组件当前是否已加载（不触发加载）。"""
        def _is_loaded(comp) -> bool:
            for attr in ("_model", "_embedder", "_converter"):
                v = getattr(comp, attr, None)
                if v is not None:
                    return True
            return False

        return {
            "parser": _is_loaded(self._parser),
            "embedder": _is_loaded(self._embedder),
            "reranker": _is_loaded(self._reranker),
        }

    def num_chunks(self) -> int:
        if not self._chunks_file.exists():
            return 0
        meta = self.load_build_meta()
        if meta and "num_chunks" in meta:
            return int(meta["num_chunks"])
        with open(self._chunks_file, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.index_ready():
            raise RuntimeError("索引不存在，请先运行: python main.py ingest（或 POST /api/ingest）")
        print("[加载] 读取本地索引 ...", flush=True)
        self._store.load(self._faiss_dir, self._embedder)
        self._bm25.load()
        self._loaded = True
        print("[加载] 索引加载完成", flush=True)
