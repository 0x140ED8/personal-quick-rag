# RAG 知识库 API 接口文档

> 版本：v1.0 | 服务定位：**纯检索服务**（解析 → 分片 → 向量化 → 混合召回 → 重排）
> LLM 生成由调用方（智能体服务）负责，本服务只返回 top-k 结构化片段。

---

## 1. 架构与调用关系

```
对话 Web 界面 ──► 智能体服务（LLM 生成） ──HTTP──► RAG 检索服务（本服务）
                                                  │
                                    POST /api/retrieve 获取相关片段
                                    GET/POST /api/config 管理配置
                                    POST /api/refresh 增量刷新索引（等价 ingest）
                                    POST /api/ingest 重建索引
```

典型对接流程：用户提问 → 智能体调 `POST /api/retrieve` 拿到片段 → 把片段拼入 prompt → LLM 基于片段生成回答。

## 2. 启动与停止

```powershell
# 启动（在 rag 项目根目录）
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py serve

# 指定监听地址/端口（覆盖 config.yaml 的 server 段）
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py serve --host 0.0.0.0 --port 8100

# 建索引（命令行方式，与 POST /api/ingest 等效）
& "C:\Users\17764\.conda\envs\rag\python.exe" -X utf8 main.py ingest
```

- 默认地址：`http://127.0.0.1:8100`（由 `config.yaml` 的 `server` 段控制）
- 交互式文档（Swagger UI）：`http://127.0.0.1:8100/docs`
- 启动时默认预热（`server.warmup: true`）：预加载索引 + embedder + reranker，约 40-60 秒，之后单次检索约 0.3-1 秒
- **停止**：Ctrl+C；服务为单进程，杀掉 python 进程即可

## 3. 端点总览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 知识库统计 |
| POST | `/api/retrieve` | **核心检索** |
| GET | `/api/config` | 查询配置（config.yaml → JSON） |
| POST | `/api/config` | 修改配置（全量替换） |
| POST | `/api/refresh` | 增量刷新索引（后台任务，等价于 /api/ingest） |
| POST | `/api/ingest` | 触发重建索引（后台任务） |
| GET | `/api/jobs/{job_id}` | 查询任务进度 |
| GET | `/api/jobs` | 列出历史任务 |

统一错误响应格式：

```json
{ "error": { "code": "HTTP_404", "message": "任务不存在: abc123" } }
```

| HTTP 状态 | code | 场景 |
|---|---|---|
| 400 | `INVALID_BODY` / `HTTP_400` | 请求体校验失败 / 配置校验失败（未知段落、字段、类型错误，message 带定位） |
| 404 | `HTTP_404` | 索引不存在、任务不存在 |
| 409 | `HTTP_409` | 索引构建中检索被拒、重复触发构建、服务忙、embedder 与索引不匹配 |
| 500 | `INTERNAL_ERROR` | 未预期异常 |

---

## 4. 检索接口（核心）

### `POST /api/retrieve`

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `query` | string | 是 | - | 用户问题，1-4096 字符 |
| `top_k` | int | 否 | `reranker.top_k`（5） | 最终返回片段数，1-50 |
| `rerank` | bool | 否 | `true` | 是否 BGE 重排；false 直接返回混合召回结果（更快） |

**请求示例**：

```bash
curl -X POST http://127.0.0.1:8100/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "STM32F103 的 ADC 有多少个通道", "top_k": 3}'
```

```javascript
// Node.js / 浏览器 fetch
const resp = await fetch("http://127.0.0.1:8100/api/retrieve", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "STM32F103 的 ADC 有多少个通道", top_k: 3 }),
});
const data = await resp.json();
```

**响应体**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 原样返回的问题 |
| `hits` | array | 片段列表，按相关性降序 |
| `hits[].chunk_id` | string | 片段唯一 ID |
| `hits[].text` | string | 片段原文 |
| `hits[].score` | float | 重排分数（rerank=true）或 RRF 融合分数（rerank=false），越大越相关 |
| `hits[].source` | string | 来源文档文件名 |
| `hits[].heading_path` | string[] | 标题层级路径，如 `["5 存储器", "5.3 寄存器描述"]` |
| `hits[].page` | int \| null | 来源页码（无页码信息时为 null） |
| `took_ms` | int | 服务端耗时（毫秒） |

**响应示例**：

```json
{
  "query": "STM32F103 的 ADC 有多少个通道",
  "hits": [
    {
      "chunk_id": "c8fced5b87c915be",
      "text": "2 个 12 位的 ADC ……",
      "score": 4.87,
      "source": "STM32F103C8T6中文数据手册1.pdf",
      "heading_path": ["5 存储器", "5.3 寄存器描述", "5.3.4 ADC 采样时间寄存器"],
      "page": 26
    }
  ],
  "took_ms": 474
}
```

**智能体拼接 prompt 推荐格式**（调用方自行实现）：

```
以下是知识库检索到的相关资料（共 {n} 段）：
[1] 来源: {source} 第{page}页 | {heading_path.join(" > ")}
{text}

请仅依据上述资料回答用户问题：{query}
```

---

## 5. 健康检查 / 统计

### `GET /api/health`

```json
{
  "status": "ok",
  "index_ready": true,
  "num_chunks": 501,
  "embedder_loaded": true,
  "reranker_loaded": true,
  "ingest_running": false,
  "uptime_sec": 2379
}
```

`status`：`ok`（索引就绪）/ `degraded`（索引不存在）。智能体侧建议启动时轮询此接口确认就绪。

### `GET /api/stats`

```json
{
  "num_chunks": 501,
  "documents": [{ "file": "STM32F103C8T6中文数据手册1.pdf", "size_bytes": 10485760 }],
  "sources": ["STM32F103C8T6中文数据手册1.pdf"],
  "components": {
    "parser": "docling",
    "chunker": "recursive",
    "embedder": "jasper (infgrad/Jasper-Token-Compression-600M)",
    "store": "faiss",
    "retriever": "hybrid",
    "reranker": "bge (BAAI/bge-reranker-v2-m3)",
    "generator": "none"
  },
  "build_meta": { "embedder_name": "jasper", "embedder_model_name": "infgrad/Jasper-Token-Compression-600M", "num_chunks": 501 }
}
```

---

## 6. 配置接口

### `GET /api/config`

把 `config.yaml` 原样转为 JSON 返回（**不含注释**）。返回内容 = 修改接口需要回传的格式。

```bash
curl http://127.0.0.1:8100/api/config
```

**响应示例**（节选）：

```json
{
  "paths": { "data_dir": "data", "index_dir": "index", "parsed_dir": "parsed" },
  "parser": { "name": "docling", "ocr": false, "table_mode": "fast", "skip_toc": true, "device": "cuda", "heading_patterns": [{ "regex": "第[一二三四五六七八九十百零〇0-9]+[卷]", "level": 1 }] },
  "chunker": { "name": "recursive", "chunk_size": 500, "chunk_overlap": 50 },
  "embedder": { "name": "jasper", "model_name": "infgrad/Jasper-Token-Compression-600M", "device": "cuda", "dtype": "bfloat16", "batch_size": 256, "compression_ratio": 0.8 },
  "store": { "name": "faiss", "distance_strategy": "MAX_INNER_PRODUCT" },
  "retriever": { "name": "hybrid", "dense_top_k": 20, "sparse_top_k": 20, "rrf_k": 60 },
  "reranker": { "name": "bge", "model_name": "BAAI/bge-reranker-v2-m3", "top_k": 5, "device": "cuda" },
  "generator": { "name": "none" },
  "server": { "host": "127.0.0.1", "port": 8100, "cors_origins": ["*"], "warmup": true }
}
```

### `POST /api/config`

**全量替换**语义：请求体必须是完整配置（建议先 GET 修改后回传）。任何未知段落/字段/类型错误返回 400（message 带具体定位）。

```javascript
// Node.js：读取 → 修改 → 回传
const cfg = await (await fetch("http://127.0.0.1:8100/api/config")).json();
cfg.reranker.top_k = 3;
cfg.retriever.dense_top_k = 30;
const resp = await fetch("http://127.0.0.1:8100/api/config", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(cfg),
});
const result = await resp.json();
console.log(result.warnings); // 注意查看警告
```

**响应体**：

```json
{
  "message": "配置已写入并生效",
  "applied": true,
  "warnings": ["已重建组件（配置段变更: ['reranker']），模型将在下次使用时懒加载"],
  "backup": "config.yaml.bak"
}
```

**生效规则**：

| 配置段变更 | 生效方式 |
|---|---|
| `retriever.*`、`reranker.top_k` | **立即生效**（下个请求生效，无需重建） |
| `parser` / `embedder` / `reranker` 的模型类参数 | 自动卸载模型 + 重建组件，下次使用时懒加载（首个请求会慢 10-30 秒） |
| `embedder.name` / `model_name` | 同上，**且现有索引失效**（向量维度/空间不匹配），检索报 409，需 `POST /api/ingest` 重建 |
| `server.host` / `port` / `warmup` | 写入文件，**重启服务后生效** |

**注意事项**：

- 写回使用 `yaml.safe_dump`，**原 config.yaml 中的注释会丢失**（自动备份为 `config.yaml.bak`）
- ingest 任务运行中禁止改配置（409）
- 组件 `name` 可选值：parser=`docling`、chunker=`recursive`、embedder=`jasper`、store=`faiss`、retriever=`hybrid`、reranker=`bge`、generator=`none`

---

## 7. 索引构建（异步任务）

### `POST /api/refresh`

`POST /api/ingest` 的等价入口（**同一套后台任务**）：执行一遍增量更新——只处理新增/变更文件，其余复用缓存。返回同样的 `202 {"job_id":...}`，进度与结果仍通过 `GET /api/jobs/{job_id}` 查询。

```bash
curl -X POST http://127.0.0.1:8100/api/refresh
# → 202 {"job_id":"a1b2c3d4e5f6","status":"running","detail":"增量刷新已开始..."}
```

- 与 `/api/ingest` 共享同一个任务槽：任一在运行时再触发（无论走哪个路径）都返回 409，不会并发构建

### `POST /api/ingest`

把 `data/` 目录下所有文件重建索引（**增量**）：对照 `index_dir/cache/manifest.json`，未变更文件直接复用解析/分片/向量缓存，只对新增/变更文件执行解析 → 切片 → 向量化；FAISS 索引仍整体重建（缓存向量 + 新向量拼接后一次建库），保存只在最后执行，失败时旧索引不被破坏。

```bash
curl -X POST http://127.0.0.1:8100/api/ingest
# → 202 {"job_id":"a1b2c3d4e5f6","status":"running","detail":"..."}
```

- 立即返回 `job_id`，构建在后台线程执行
- 重复触发：409（已有任务运行）
- 构建期间：检索请求返回 409
- 换 embedder 配置后必须重新 ingest，否则检索报 409（索引兼容性校验）

### `GET /api/jobs/{job_id}`

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "stage": "embedding",
  "done": 320,
  "total": 501,
  "detail": "",
  "created_at": 1759000000.0,
  "updated_at": 1759000123.4,
  "result": null,
  "error": null,
  "logs": ["[3.向量化] 正在编码 501 个 chunk ...", "..."]
}
```

| 字段 | 说明 |
|---|---|
| `status` | `queued` / `running` / `done` / `failed` |
| `stage` | `parsing` → `chunking` → `embedding` → `saving` → `done` |
| `done` / `total` | 阶段内进度（如第 3/10 个文件、第 320/501 个 chunk） |
| `detail` | 阶段细节（如当前解析的文件名） |
| `result` | 构建成功后的统计：`{"chunks": 501, "files": 1, "failed_files": [], "took_sec": 120.5, "files_reused": 0, "files_added": 1, "files_removed": 0, "chunks_reused": 0, "chunks_embedded": 501}`。`files_reused/chunks_reused` = 复用缓存的量；`files_added/chunks_embedded` = 本次实际解析/编码的量；`files_removed` = 已从磁盘删除、缓存被清理的文件数 |
| `logs` | 捕获的构建日志尾部（最近 50 行），可用于前端展示进度 |

### `GET /api/jobs`

列出最近 20 个任务：`{"jobs": [{"job_id": "...", "status": "done", "created_at": "..."}]}`

**前端轮询示例**：

```javascript
const { job_id } = await (await fetch("http://127.0.0.1:8100/api/ingest", { method: "POST" })).json();
const timer = setInterval(async () => {
  const job = await (await fetch(`http://127.0.0.1:8100/api/jobs/${job_id}`)).json();
  console.log(job.stage, `${job.done}/${job.total}`);
  if (job.status === "done" || job.status === "failed") clearInterval(timer);
}, 3000);
```

---

## 8. 对接注意事项

1. **单并发**：GPU 推理不宜并发，服务端同一时刻只处理一个检索请求，并发请求返回 409，调用方应捕获后重试。
2. **首次请求慢**：若服务刚启动且 `warmup: false`，或刚改过模型类配置，首个检索请求需加载模型（10-30 秒），属正常现象。
3. **CORS**：默认 `["*"]` 全放开（本地部署）；跨域部署时在 `server.cors_origins` 配置具体来源。
4. **PowerShell 5.1 测试注意**：`Invoke-RestMethod` 发中文需 `[Text.Encoding]::UTF8.GetBytes($json)` 作为 `-Body`，否则乱码（服务端响应已带 `charset=utf-8`，回读无碍）。
5. **改 embedder 必须重建索引**：不同模型向量维度不同（Jasper 2048 维），FAISS 索引不通用。服务端有兼容性校验（`index/build_meta.json`），不匹配时返回 409 而非错误结果。
6. **`data/` 目录**：ingest 只处理 `data/` 下的**文件**（不递归子目录），新增文档后需重新 `POST /api/ingest`。重建是增量的：未变更文件复用 `index_dir/cache/` 缓存，只有新增/变更文件会重新解析与向量化；`parser`/`chunker`/`embedder`（含 dtype、compression_ratio）参数变化会使对应缓存失效并触发重建。
7. **显存约束（8GB 显卡）**：服务模式下 embedder（约 1.2GB）+ reranker（约 2.3GB）常驻；ingest 时会先卸载它们再加载解析/编码模型，全程不超 8GB。

## 9. 配置参考（config.yaml 全字段）

见 `GET /api/config` 响应示例（第 6 节）。关键可调参数：

| 参数 | 位置 | 作用 |
|---|---|---|
| `retriever.dense_top_k` / `sparse_top_k` | 召回 | 双路召回候选数（影响召回覆盖与耗时） |
| `retriever.rrf_k` | 召回 | RRF 融合常数（一般不动） |
| `reranker.top_k` | 重排 | 默认返回片段数 |
| `chunker.chunk_size` / `chunk_overlap` | 分片 | 切片大小/重叠（改后需重新 ingest） |
| `embedder.batch_size` | 向量化 | 编码批大小（OOM 则调小） |
| `parser.device` / `embedder.device` / `reranker.device` | 各阶段 | `auto` / `cuda` / `cpu` |
| `server.host` / `port` / `cors_origins` / `warmup` | 服务 | 监听与跨域配置 |
