# RAG · 本地知识库检索服务

一个**插件式、纯检索**的 RAG 系统：`文档解析 → 分片 → 向量化 → 混合召回 → 重排`，对外提供 HTTP 检索 API 供智能体 / Web 调用（LLM 生成由调用方负责）。解析用 **Docling**，向量化用 **Jasper-Token-Compression-600M**，召回为 **FAISS 稠密 + BM25 稀疏 + RRF 融合**，重排用 **BAAI/bge-reranker-v2-m3**；全流程本地离线推理，支持按文件增量建索引，8GB 显存即可运行。

## 功能特性

### 检索流水线

- **解析（Docling）**：PDF/Word/txt等多种格式 → Markdown，`table_mode=fast`，支持 OCR 开关与自定义标题正则（如 `第X卷 / 第X章`）提取标题层级
- **分片（Recursive）**：按 `chunk_size / chunk_overlap` 递归切分，保留 `heading_path` 与页码
- **向量化（Jasper-600M）**：SentenceTransformer 直连封装，query 侧带指令前缀、支持 token 压缩率调节，2048 维
- **存储（FAISS）**：`MAX_INNER_PRODUCT` 内积检索，`chunks.jsonl` 与索引分离保存
- **混合召回（Hybrid）**：稠密 + BM25（jieba 中文分词）双路召回，RRF 融合排序
- **重排（BGE）**：bge-reranker-v2-m3 精排，可 `rerank=false` 跳过以加速
- **生成（占位）**：`none` 生成器，服务定位为纯检索，LLM 由智能体侧拼接 prompt 完成

### 增量建索引

- 对照 `index_dir/cache/manifest.json` 逐文件比对，未变更文件复用缓存（跳过解析/切片/向量化），新增/变更文件走完整流程，已删除文件清缓存
- 缓存 key = `sha256(文件名|mtime|size|parser指纹|chunker指纹|embedder指纹)`，只纳入影响索引的参数；纯复用时不加载 embedder
- FAISS 仍用「缓存向量 + 新向量」整体重建，**保存只在最后**，失败不破坏旧索引
- 构建结果含 `files_reused / files_added / files_removed / chunks_reused / chunks_embedded` 统计

### API 服务

- FastAPI + uvicorn 单 worker，`GET /api/health`、`/api/stats`、`/api/config`、`/api/jobs*`，`POST /api/retrieve`、`/api/ingest`、`/api/refresh`、`/api/config`
- `POST /api/ingest` / `/api/refresh` 后台异步任务：立即返回 `job_id`，`GET /api/jobs/{job_id}` 轮询阶段进度（parsing → chunking → embedding → saving）
- 索引构建中检索返回 409，单 worker + Lock 保证 GPU 推理串行
- 检索与 ingest 互斥、不排队；`build_meta.json` 指纹校验，换 embedder 后检索 409 提示重建

### 工程化

- **registry 插件式架构**：新增组件 = 新文件 + `__init__.py` 注册 + 配置切换，不改管线代码
- 配置为 dataclass + YAML（`config.yaml`），`POST /api/config` 全量替换并热生效（模型类参数自动重建、懒加载）
- `main.py` 启动前依赖自检、强制 HF 离线、Windows 控制台 GBK 容错，Ctrl+C 强制退出

## 技术栈

| 层         | 技术                                                          | 版本                       |
| ---------- | ------------------------------------------------------------- | -------------------------- |
| 语言       | Python（conda `rag` 环境）                                    | 3.9                        |
| 文档解析   | Docling                                                       | 2.69.1                     |
| 分片       | langchain-text-splitters（RecursiveCharacterTextSplitter）    | 0.3.11                     |
| 向量化     | sentence-transformers + Jasper-Token-Compression-600M         | 2048 维 / bf16             |
| 向量存储   | FAISS（faiss-cpu）                                            | 1.11.0                     |
| 稀疏召回   | rank-bm25 + jieba                                             | 0.2.2 / 0.42.1             |
| 重排       | BAAI/bge-reranker-v2-m3（transformers）                       | —                          |
| 推理       | PyTorch + CUDA（torch/transformers）                          | —                          |
| API 服务   | FastAPI + uvicorn                                             | —                          |
| 配置       | PyYAML（dataclass 映射 + 校验）                               | —                          |

## 目录结构

```
rag/
├── main.py                      # CLI 入口：ingest / query / serve，含依赖自检与控制台容错
├── config.yaml                  # 运行配置（由 POST /api/config 全量生成，注释会丢失）
├── requirements.txt             # rag 环境完整依赖清单
├── src/
│   ├── config.py                # dataclass 配置定义 + YAML 加载/校验
│   ├── registry.py              # 阶段名 → 组件实现的注册表
│   ├── pipeline.py              # RAGPipeline：装配组件，ingest / retrieve / query
│   ├── cache.py                 # 按文件增量缓存（manifest + 指纹计算）
│   ├── types.py / utils.py      # Chunk / Hit 数据结构、进度条等
│   ├── parsing/                 # docling_parser、heading_extractor（标题层级提取）
│   ├── chunking/                # recursive_chunker
│   ├── embedding/               # jasper_embedder（SentenceTransformer 直连 + FAISS 适配器）
│   ├── storage/                 # faiss_store
│   ├── retrieval/               # vector_retriever、bm25_retriever、hybrid_retriever（RRF）
│   ├── reranking/               # bge_reranker
│   ├── generation/              # none_generator（占位，供调用方自行生成）
│   └── api/                     # app.py（应用工厂）、schemas.py（请求/响应模型）
├── data/                        # 待索引文档（只处理直接文件，不递归子目录）
├── index/                       # FAISS 索引、chunks.jsonl、build_meta.json、cache/
├── parsed/                      # 解析产物 Markdown（docling 输出）
├── test/                        # 手工实验脚本（非流水线入口）
├── RAG构建指南.md               # RAG 原理与选型教程
├── API接口文档.md               # 完整接口文档（端点、示例、错误码）
└── AGENTS.md                    # 项目须知（环境、命令、陷阱）
```

## 环境准备

| 软件         | 要求        | 说明                                                                   |
| ------------ | ----------- | ---------------------------------------------------------------------- |
| Python       | **3.9**     | 必须用 conda `rag` 环境解释器（勿用 base，其 torch DLL 会加载失败）    |
| CUDA GPU     | 8GB 显存    | 解析/编码/重排均走 GPU；无显卡可把各 `device` 改为 `cpu`（慢）         |
| HuggingFace  | 本地缓存    | 模型全部走本地缓存，代码强制 `HF_HUB_OFFLINE=1`，不联网下载            |
| Windows 控制台 | GBK       | 打印中文已做容错；独立脚本需自行处理编码                               |

```powershell
conda create -n rag python=3.9
conda activate rag
pip install -r requirements.txt
```

## 快速开始

启动顺序：**准备文档 → 建索引 → 检索 / 起服务**。所有命令在项目根目录执行。

### 1. 放置文档

把待索引文件放入 `data/`（**只处理直接文件，不递归子目录**；`data/_other/` 会被静默跳过）。

### 2. 建立索引

```powershell
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py ingest
```

- 增量执行：未变更文件直接复用 `index/cache/` 缓存，仅新增/变更文件解析与向量化
- 首次 ingest 视文档量需数分钟；进度按 `parsing → chunking → embedding → saving` 输出

### 3. 命令行提问（可选）

```powershell
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py query "STM32F103 的 ADC 有多少个通道"
```

检索 + BGE 重排后打印命中片段（CLI 结束后自动卸载 embedder / reranker 释放显存）。

### 4. 启动 API 服务（端口 8100）

```powershell
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py serve
# 可覆盖监听地址：main.py serve --host 0.0.0.0 --port 8100
```

- 地址：`http://127.0.0.1:8100`，交互式文档：`http://127.0.0.1:8100/docs`
- `server.warmup: true` 时启动预热索引 + embedder + reranker，约 40-60 秒（`/api/health` 的 `embedder_loaded`、`reranker_loaded` 均为 `true` 才就绪）
- 就绪后单次检索约 0.3-1 秒；停止：Ctrl+C（已做强制退出处理）

```powershell
# 检索示例（PowerShell 5.1 发中文 JSON 必须按 UTF-8 字节作 Body）
$json = '{"query":"STM32F103 的 ADC 有多少个通道","top_k":3}'
Invoke-RestMethod -Uri http://127.0.0.1:8100/api/retrieve -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($json))
```

完整接口说明见 `API接口文档.md`（端点总览、错误码、前端轮询示例）。

## 索引与缓存机制

- `index/build_meta.json` 记录构建时的 parser / chunker / embedder 指纹与 `num_chunks`；检索前校验，**换 embedder（或改任何进入指纹的参数）后必须重新 ingest**，否则返回 409
- `index/cache/manifest.json` 为 `{"文件名": "key"}`；`cache/<key>/` 存该文件的 `sections.json`、`chunks.json`、`vectors.npz`
- 不进指纹的参数（device、batch_size、目录路径、retriever / reranker 配置）可随意改动，不会导致缓存失效
- 增量过程失败不会破坏旧索引（FAISS 保存只在最后一步）

## 配置项

`config.yaml` 由 `POST /api/config` 全量生成（原注释丢失、自动备份 `config.yaml.bak`），关键段落：

- `paths.data_dir / index_dir / parsed_dir`：**指向当前激活的知识库**（外部目录），改路径前先读 `config.yaml` 确认工作目标
- `parser.*`：`name`（docling）、`ocr`、`table_mode`（CUDA 上务必 `fast`）、`skip_toc`、`device`、`heading_patterns`
- `chunker.*`：`chunk_size` / `chunk_overlap`
- `embedder.*`：`name`（jasper）、`model_name`、`device`、`dtype`（bfloat16）、`batch_size`（OOM 调小）、`compression_ratio`
- `retriever.*`：`dense_top_k` / `sparse_top_k` / `rrf_k`
- `reranker.*`：`name`（bge）、`model_name`、`top_k`、`device`
- `server.*`：`host` / `port` / `cors_origins` / `warmup`

改配置的三种方式：直接编辑 YAML（重启生效）、`POST /api/config`（热生效，检索参数下个请求即生效）、CLI 参数覆盖 `--host/--port`。新增参数需同步三处：dataclass 字段、`pipeline.py` 的 registry.create 调用、`config.yaml`。

## 显存约束（8GB 笔记本显卡）

- 大模型阶段用完即卸：解析后卸 parser、ingest 后卸 embedder、CLI query 后卸 embedder + reranker
- **例外：serve 模式模型常驻**（embedder 约 1.2GB + reranker 约 2.3GB）；但 ingest 结束时仍会卸 embedder，重建索引后的首次检索需重新加载（慢）
- Jasper batch_size=256 峰值约 3.2GB；Docling 的 `table_mode=accurate` 在 CUDA 上极易卡死，保持 `fast`

## 与智能体 / 前端对接

```
对话 Web（pi-agent-web） ──► 智能体服务（pi-agent-server，LLM 生成） ──HTTP──► RAG 检索服务（本服务 :8100）
```

- 智能体调 `POST /api/retrieve` 拿片段 → 拼入 prompt → LLM 生成回答；推荐拼接格式见 `API接口文档.md`
- 知识库管理页面通过 pi-agent-server 的 `/api/kb/*` 代理本服务的 config / ingest / jobs 接口
- 服务端单并发：并发检索返回 409，调用方应捕获后重试；跨域部署在 `server.cors_origins` 配置具体来源（默认 `["*"]`）

## 注意事项

- **换 embedder（或任何进入指纹的参数）后必须重新 ingest**：向量空间不同，FAISS 不通用，服务端以 409 拦截
- 新增文档后需重新 ingest（`data/` 不递归子目录）；增量模式下旧文档零成本复用
- Jasper 不能用 LangChain HuggingFaceEmbeddings 封装：query 侧需要 `prompt_name="query"` 与 `compression_ratio`，只能直封 SentenceTransformer（见 `src/embedding/jasper_embedder.py`）
- `test/` 下脚本为手工实验（bge / docling / jasper / Qwen3 等），不属于流水线入口
- 无正式测试 / lint，验证方式：`py_compile` 关键文件 + 临时目录冒烟脚本；改完建议跑一次 `main.py ingest` 确认索引可重建
