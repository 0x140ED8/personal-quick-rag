"""命令行入口。

用法：
    python main.py ingest                  # 建立索引
    python main.py query "你的问题"         # 检索 + 重排 + 生成
    python main.py serve                   # 启动 API 检索服务（供智能体/web 调用）
"""
from __future__ import annotations

import argparse
import os
import sys

# 模型均已本地缓存，强制离线加载，避免访问 huggingface.co 失败导致长时间重试
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _setup_console() -> None:
    """让 Windows 控制台输出更健壮：不可编码字符（如 ®）替换为 ?，避免 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _check_deps() -> None:
    """启动前检查关键依赖，给出可操作的提示（常见于用错了 Python 环境）。"""
    missing = []
    for mod, pkg in (
        ("jieba", "jieba"),
        ("docling", "docling"),
        ("rank_bm25", "rank-bm25"),
        ("langchain_huggingface", "langchain-huggingface"),
        ("faiss", "faiss-cpu"),
    ):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
    if missing:
        print(f"[启动检查] 当前 Python: {sys.executable}", flush=True)
        print(f"[启动检查] 缺少依赖: {' '.join(missing)}", flush=True)
        print("[启动检查] 请先激活 rag 环境并安装依赖，例如：", flush=True)
        print("    conda activate rag", flush=True)
        print(f"    pip install {' '.join(missing)}", flush=True)
        sys.exit(1)


def main() -> None:
    _setup_console()
    parser = argparse.ArgumentParser(description="模块化 RAG")
    parser.add_argument(
        "command", choices=["ingest", "query", "serve"], help="ingest=建索引 / query=提问 / serve=启动API服务"
    )
    parser.add_argument("text", nargs="*", help="query 命令的问题内容")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--host", default=None, help="serve: 覆盖配置的监听地址")
    parser.add_argument("--port", type=int, default=None, help="serve: 覆盖配置的监听端口")
    args = parser.parse_args()

    # 依赖检查放在导入重依赖之前，避免 ModuleNotFoundError 的堆栈难以定位
    _check_deps()

    from src.config import load_config
    from src.registry import registry
    from src.pipeline import RAGPipeline

    # 触发各子包注册组件
    import src.parsing  # noqa: F401
    import src.chunking  # noqa: F401
    import src.embedding  # noqa: F401
    import src.storage  # noqa: F401
    import src.reranking  # noqa: F401
    import src.generation  # noqa: F401

    print(f"[启动] Python: {sys.executable}", flush=True)

    if args.command == "serve":
        _serve(args)
        return

    config = load_config(args.config)
    pipeline = RAGPipeline(config, registry)

    if args.command == "ingest":
        pipeline.ingest()
    else:
        query = " ".join(args.text).strip()
        if not query:
            print("请提供问题：python main.py query \"你的问题\"")
            return
        print(f"问题: {query}\n")
        print(pipeline.query(query))


def _serve(args) -> None:
    """启动 FastAPI 检索服务（uvicorn 单 worker）。"""
    from src.api.app import create_app

    app = create_app(args.config)

    # 命令行参数覆盖配置（配置文件里的 server.host/port 仍为默认来源）
    from src.config import load_config

    server_cfg = load_config(args.config).server
    host = args.host or server_cfg.host
    port = args.port or server_cfg.port

    import uvicorn

    print(f"[服务] 启动 API: http://{host}:{port}（交互式文档: /docs）", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Docling 的流水线使用非 daemon 线程，线程阻塞在 CUDA 算子时普通退出无法结束进程，
        # 这里强制终止，保证 Ctrl+C 一定能退出。
        print("\n[中断] 收到 Ctrl+C，正在强制退出 ...", flush=True)
        os._exit(130)
