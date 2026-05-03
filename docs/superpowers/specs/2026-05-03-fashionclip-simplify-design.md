# FashionCLIP API 精简设计

## 背景

fashionclip-api 原为包含图片向量化、数据库存储、相似搜索的完整服务。随着 `ht-files` 项目接管了文件扫描和数据库操作，fashionclip-api 需要精简为纯无状态的 CLIP embedding 服务。

## 架构

```
ht-files ── POST /embed ────────→ fashionclip-api ──→ CLIP 模型 ──→ embedding
         ── POST /embed-batch ──→               ──→ CLIP 模型 ──→ embeddings[]
         ── GET  /health ────────→               ──→ 200 OK
```

- 无状态：不连接任何数据库，不持久化任何数据
- 每次请求从本地文件系统或 HTTP URL 读取图片 → 向量化 → 返回结果
- ht-files 负责所有持久化（扫描文件列表、存向量到 PostgreSQL）

## API 端点

### POST /embed — 单图向量化

请求:
```json
// 本地文件模式
{"base": "dapai-s", "path": "2026年/3月/0331/img.jpg"}
// HTTP URL 模式
{"base": "", "path": "http://100.64.0.6:5244/dapai-s/path/to/img.jpg"}
```

响应:
```json
{"embedding": [0.12, -0.34, ...]}
```

路径映射规则：
- `base` 非空 → 拼接为 `/mnt/{base}/{path}`，直接从本地文件系统读取
- `base` 为空 → `path` 作为 HTTP URL 下载

错误处理：
- `base` 和 `path` 必填，否则返回 400
- 文件不存在 / 下载失败返回 400
- 图片无法解析返回 400
- 模型未加载返回 503

### POST /embed-batch — 批量向量化

请求:
```json
// 本地文件模式
{"base": "dapai-s", "paths": ["2026年/3月/0331/img1.jpg", "2026年/3月/0331/img2.jpg"]}
// HTTP URL 模式
{"base": "", "paths": ["http://100.64.0.6/1.jpg", "http://100.64.0.6/2.jpg"]}
```

响应:
```json
{
  "embeddings": [[0.12, ...], [-0.05, ...]],
  "errors": []
}
```

- `paths` 最多 20 个，超限返回 400
- 本地模式：使用 `ThreadPoolExecutor` 并行读取，上限 `min(8, len(paths))`
- URL 模式：使用 `ThreadPoolExecutor` 并行下载
- 失败项记录到 `errors` 列表，不整批失败

### GET /health — 健康检查

响应:
```json
{"status": "ok", "model_loaded": true, "device": "cuda"}
```

- 不依赖数据库，仅确认模型已加载

## 数据流

### 本地文件模式（主要路径）

1. ht-files 扫描目录，获得文件相对路径
2. 调用 `POST /embed-batch` 或 `POST /embed`，传入 `base + paths`
3. fashionclip-api 拼接为 `/mnt/{base}/{path}`，直接从磁盘读取图片
4. CLIP 编码 → L2 归一化 → 返回 512 维向量
5. ht-files upsert 到 PostgreSQL

### HTTP URL 模式（备用路径）

1. ht-files 传入完整 HTTP URL
2. fashionclip-api 下载图片 → 向量化 → 返回
3. 用于非本地挂载的图片来源

## 配置

### 保留的变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` (auto-detect) | 运行设备，无 CUDA 自动 fallback 到 CPU |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用 ffmpeg 图片缩放 |
| `FASHIONCLIP_MAX_DIM` | `672` | 缩放后最大尺寸 (px) |
| `FASHIONCLIP_QUALITY` | `85` | JPEG 压缩质量 |

### 删除的变量

- `PHOTOS`, `IMAGE_BASE_URL`, `ALIST_BASE_URL`, `SCAN_ROOTS` — 文件扫描相关
- `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY` — 数据库相关
- `REDIS_URL`, `BATCH_SIZE` — 扫描任务队列

## 删除的功能模块

| 模块 | 代码 |
|------|------|
| Qdrant 初始化与连接 | `init_qdrant()`, `qdrant_client` |
| 文件扫描 | `scan_photos_directory()`, `is_valid_image()`, `image_to_url()` |
| Checkpoint 系统 | `save_checkpoint()`, `load_checkpoint()`, `clear_checkpoint()` |
| 异步扫描任务 | `/embed/scan`, `/embed/cancel`, `/embed/status` |
| 相似搜索 | `/search` |
| 向量信息查询 | `/embed/{path}` |
| 旧批量上传 | `/embed/batch` (上传文件版) |
| SCAN_ROOTS 解析 | `_parse_scan_roots()` |
| 信号处理 | SIGTERM/SIGINT handler |

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `app.py` | 重写 | 从 ~950 行缩到 ~200 行 |
| `requirements.txt` | 删除 qdrant-client | 只保留 web + CLIP + 图片库 |
| `Dockerfile` | 简化 | GPU 可选，支持 CPU 模式 |
| `docker-compose.yml` | 简化 | 删除 Qdrant 服务依赖 |
| `CLAUDE.md` | 更新 | 同步项目描述和命令 |
| `API.md` | 更新 | 只保留 3 个端点 |
| `SPEC.md` | 更新 | 同步精简后的规格 |
| `.env.example` | 更新 | 只保留有效变量 |
| `.env.docker` | 更新 | 只保留有效变量 |
| `entrypoint.sh` | 保留 | 通常不需要更改 |

## Docker 部署

### GPU 模式
```bash
docker run -d --gpus all -p 8008:8008 fashionclip-api
```

### CPU 模式
```bash
docker run -d -p 8008:8008 -e DEVICE=cpu fashionclip-api
```
