# RAG 项目须知

> 系统全局指令里的 RTK 块不适用于本机：`rtk` 未安装（不在 PATH，`rtk ...` 必然 command not found），直接用原生命令。

## 运行环境（Windows 特有，容易踩坑）

- 必须用 conda `rag` 环境的解释器，直接调 python.exe（勿用 base conda：其用户 site-packages 的 torch DLL 加载会失败）：
  ```powershell
  & "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 ...
  ```
- 避免 `conda run -n rag ...`：它用 GBK 回显子进程输出，非 ASCII 字符会 UnicodeEncodeError 崩溃。
- 控制台是 GBK：新代码打印中文/特殊符号前需 `stream.reconfigure(errors="replace")` 或只用 ASCII（参考 main.py `_setup_console`、src/utils.py `ProgressBar`）。
- 模型全部走本地 HuggingFace 缓存。main.py 已设 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`；独立脚本需自行设置，否则长时间重试网络。

## 常用命令与验证

```powershell
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py ingest        # 建索引（增量）
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py query "问题"   # 检索+重排
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py serve         # API 服务（默认 :8100）
```

- 无正式测试/lint/typecheck。验证 = `py_compile` + 临时目录冒烟脚本：
  ```powershell
  & "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 -m py_compile main.py src\config.py src\pipeline.py src\api\app.py
  $env:PYTHONPATH="D:\MyAPP\rag"; & "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 <smoke_script.py>   # import src.* 的独立脚本需 PYTHONPATH
  ```
- API 冒烟：`serve` 预热约 40-60 秒（/api/health 的 embedder_loaded/reranker_loaded 均 true 才就绪）再打 /api/retrieve。PowerShell 5.1 发中文 JSON 必须 `[Text.Encoding]::UTF8.GetBytes($json)` 作 `-Body`，否则乱码。
- 依赖清单：requirements.txt。

## 重要：config.yaml 指向哪个知识库

- `config.yaml` 由 `POST /api/config` 全量生成（注释丢失、自动 `.bak`），`paths.*` 指向**当前激活的外部知识库**（现为 `D:\MyAPP\agent\pi-agent-server\data\knowledge-bases\<id>\`）。
- 因此 `main.py ingest` 操作的是那个活动 KB，不是本仓库的 `data/`、`index/`（后者是旧实验残留，勿当默认工作目录）。改索引/路径前先读 config.yaml 确认。

## 架构（registry 插件式流水线）

- main.py → load_config(config.yaml) → RAGPipeline(src/pipeline.py)，按配置从 registry 装配：parser(docling) → chunker → embedder → store(faiss) → retriever(hybrid: dense+BM25+RRF) → reranker(bge) → generator(占位)。
- 新增组件 = `src/<stage>/` 新文件 + 在 `src/<stage>/__init__.py` 注册 + config.yaml 切换，不改管线代码。
- 配置是 dataclass + YAML（src/config.py）；新参数需同步三处：dataclass 字段、pipeline.py 的 registry.create 调用、config.yaml。
- ingest 是**增量**：比对 `index_dir/cache/manifest.json`（`{"文件名": "key"}`），未变更文件复用 `cache/<key>/{sections.json,chunks.json,vectors.npz}`（跳过解析/切片/向量化，纯复用时不加载 embedder）；新增/变更文件走完整流程；已删除文件清缓存。FAISS 仍用「缓存向量 + 新向量」整体重建，save 只在最后（失败不破坏旧索引）。result 含 files_reused/files_added/files_removed/chunks_reused/chunks_embedded。
- 缓存 key = sha256(文件名|mtime_ns|size|parser指纹|chunker指纹|embedder指纹)；`src/cache.py` 只纳入影响最终索引的参数（排除 device/batch_size/目录等），改 retriever/reranker 等不会失效。build_meta.json 存同一套指纹，检索前校验不符返回 409。
- ingest 只处理 `data/` 下的**直接文件，不递归子目录**（如 `data/_other/` 静默跳过）。新增文档后需重新 ingest。

## API 服务层（src/api/，main.py serve）

- FastAPI 应用工厂 `src/api/app.py:create_app()`；请求/响应模型在 src/api/schemas.py；接口文档见 `API接口文档.md`。
- 端点：GET /api/health、/api/stats、/api/config、/api/jobs；POST /api/retrieve、/api/config、/api/ingest、/api/refresh（与 /api/ingest 等价的后台增量刷新，共用同一任务槽）。
- 单 worker + `threading.Lock`（**必须 Lock 不能 RLock**：ingest 请求线程获取、后台线程释放）。检索与 ingest 互斥，不排队直接 409。
- 服务模式模型常驻（不走 pipeline.query() 的 unload）；但 ingest 结束时仍会 unload embedder（reranker 保持常驻），故重建索引后的首次检索需重新加载 embedder（慢）。
- `POST /api/config`：全量替换；除 `server` 外的配置段变更都会 rebuild_pipeline()（先 unload 再重建，懒加载），检索参数下个请求即生效；换 embedder 后索引失效，需重新 ingest。
- 所有 JSON 响应走 `UTF8JSONResponse`（显式 charset=utf-8）：PowerShell 5.1 无 charset 时按 Latin-1 解码，曾导致 config 回读回写把中文 regex 损坏。
- 前端在另一个仓库 `D:\MyAPP\agent\pi-agent-web`（KnowledgePanel.tsx 调 pi-agent-server 的 /api/kb/*，后者代理到本服务）；改 ingest 展示/i18n 去那里，验证用 `npm run typecheck`。

## 显存约束（8GB 笔记本显卡）

- 大模型阶段用完即卸：解析后卸 parser、ingest 后卸 embedder、CLI query 后卸 embedder+reranker——新增占显存组件时保持该模式。**例外：serve 模式模型常驻。**
- Jasper-600M bf16 约 1.2GB，batch_size=256 峰值约 3.2GB；BGE reranker 约 2.3GB。
- Docling 的 table_mode=accurate 在 CUDA 上极易卡死，用 fast。

## 关键陷阱

- **换 embedder（或任何进入指纹的参数）后必须重新 ingest**：维度/向量空间不同，FAISS 不通用；服务端用 build_meta.json 拦截，不匹配时 /api/retrieve 返回 409。
- Jasper 不能用 LangChain HuggingFaceEmbeddings 封装：它需要 query 侧 `prompt_name="query"` 和 `compression_ratio`，只能直接封装 SentenceTransformer（见 src/embedding/jasper_embedder.py，含给 FAISS 用的适配器）。
- `test/` 下的 bge-reranker-v2-m3.py / docling.py / Jasper-Token-Compression-600M.py / Qwen3-Embedding-4B.py 等是手工实验脚本，不属于流水线，勿当作入口。
- Youtu-Embedding 已移除（代码/配置/文档），embedder 仅剩 jasper；registry 中不存在 "youtu" 键。


<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->
