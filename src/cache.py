"""按文件粒度的增量索引缓存。

布局（index_dir/cache/）：
- manifest.json: {"文件名": "<key>"}，每个知识库（index_dir）一份
- <key>/sections.json: 解析缓存（Section 列表）
- <key>/chunks.json:   分片缓存（Chunk 列表，复用时不跑 chunker）
- <key>/vectors.npz:   vectors(float32 [n, d]) + chunk_ids（与分片对齐校验）

key = sha256(文件名 | mtime_ns | size | parser指纹 | chunker指纹 | embedder指纹)：
只纳入「会影响 sections/chunks/vectors 最终结果」的参数
（device / batch_size / 各目录路径等不影响索引的参数不入 key），
任何一项变化都会使该文件缓存失效。

写盘采用 tmp + os.replace（目录与 manifest 都原子替换），
ingest 中断不会留下半截缓存。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.types import Chunk, Section


def _hash_json(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class FileCacheManager:
    def __init__(self, cache_dir) -> None:
        self._dir = Path(cache_dir)
        self._manifest_file = self._dir / "manifest.json"
        self._manifest: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 参数指纹与 key
    # ------------------------------------------------------------------ #
    @staticmethod
    def fingerprints(parser_cfg, chunker_cfg, embedder_cfg) -> dict:
        """影响最终索引的参数指纹；排除 device/batch_size/目录等无关项。"""
        return {
            "parser": _hash_json(
                {
                    "name": parser_cfg.name,
                    "ocr": parser_cfg.ocr,
                    "table_mode": parser_cfg.table_mode,
                    "skip_toc": parser_cfg.skip_toc,
                    "heading_patterns": parser_cfg.heading_patterns,
                }
            ),
            "chunker": _hash_json(
                {
                    "name": chunker_cfg.name,
                    "chunk_size": chunker_cfg.chunk_size,
                    "chunk_overlap": chunker_cfg.chunk_overlap,
                }
            ),
            "embedder": _hash_json(
                {
                    "name": embedder_cfg.name,
                    "model_name": embedder_cfg.model_name,
                    "dtype": embedder_cfg.dtype,
                    "compression_ratio": embedder_cfg.compression_ratio,
                }
            ),
        }

    @staticmethod
    def key_for(path, fingerprints: dict) -> str:
        p = Path(path)
        st = p.stat()
        raw = "|".join(
            [
                p.name,
                str(st.st_mtime_ns),
                str(st.st_size),
                fingerprints["parser"],
                fingerprints["chunker"],
                fingerprints["embedder"],
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # manifest 读写
    # ------------------------------------------------------------------ #
    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

    def load(self) -> None:
        self._manifest = {}
        if not self._manifest_file.exists():
            return
        try:
            data = json.loads(self._manifest_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._manifest = {str(k): str(v) for k, v in data.items()}
        except Exception as e:  # noqa: BLE001
            print(f"[缓存] manifest 读取失败，按空缓存处理: {e}", flush=True)

    def _save_manifest(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        _write_json(self._manifest_file, self._manifest)

    # ------------------------------------------------------------------ #
    # 缓存条目读写
    # ------------------------------------------------------------------ #
    def read(self, name: str):
        """读取某个文件的缓存；任何异常（缺失/损坏/不对齐）返回 None（视为失效）。"""
        key = self._manifest.get(name)
        if not key:
            return None
        entry = self._dir / key
        try:
            sections = [
                Section.from_dict(d)
                for d in json.loads((entry / "sections.json").read_text(encoding="utf-8"))
            ]
            chunks = [
                Chunk.from_dict(d)
                for d in json.loads((entry / "chunks.json").read_text(encoding="utf-8"))
            ]
            with np.load(entry / "vectors.npz", allow_pickle=False) as npz:
                vectors = npz["vectors"].astype(np.float32)
                chunk_ids = [str(x) for x in npz["chunk_ids"]]
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(chunks)
                or chunk_ids != [c.chunk_id for c in chunks]
            ):
                raise ValueError("向量与分片不对齐")
            return sections, chunks, vectors
        except Exception as e:  # noqa: BLE001
            print(f"[缓存] {name} 缓存不可用（{e}），将重新处理", flush=True)
            return None

    def write(self, name: str, key: str, sections: list, chunks: list, vectors) -> None:
        """原子写缓存条目并更新 manifest（先写 tmp 目录，再整体 rename）。"""
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / f"{key}.tmp-{os.getpid()}"
        final = self._dir / key
        try:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            _write_json(tmp / "sections.json", [s.to_dict() for s in sections])
            _write_json(tmp / "chunks.json", [c.to_dict() for c in chunks])
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.size == 0:
                arr = arr.reshape(0, 0)
            np.savez(
                tmp / "vectors.npz",
                vectors=arr,
                chunk_ids=np.asarray([c.chunk_id for c in chunks], dtype="U32"),
            )
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._manifest[name] = key
        self._save_manifest()

    def delete(self, name: str) -> None:
        key = self._manifest.pop(name, None)
        if key:
            shutil.rmtree(self._dir / key, ignore_errors=True)
            self._save_manifest()

    def prune_orphans(self) -> int:
        """清理 manifest 未引用的目录/文件（中断遗留的 tmp 等）。"""
        if not self._dir.exists():
            return 0
        valid = set(self._manifest.values())
        removed = 0
        for p in self._dir.iterdir():
            if p.name == "manifest.json" or p.name in valid:
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        return removed
