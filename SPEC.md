# FashionCLIP 图片相似度搜索 API 服务

## 概述

基于 FashionCLIP 模型的本地图片向量化和相似图片搜索服务。为服饰鞋帽行业提供专门的图像特征提取，支持按需 embedding 和相似图搜索。

## 技术栈

- **运行时**: Python 3.10 + CUDA 12.1
- **框架**: FastAPI
- **模型**: FashionCLIP (laion/CLIP-ViT-B-16-laion2B-s34B-b88K)
- **向量数据库**: Qdrant (可选，用于持久化存储和高效检索)
- **镜像**: nvidia/cuda:12.1.0-runtime-ubuntu22.04

## 核心功能

### 1. 图片 Embedding

- 扫描配置目录下所有图片文件
- 支持的图片格式: JPEG, JPG, PNG
- 过滤规则:
  - 跳过 `._` 开头的 macOS 临时文件
  - 跳过文件名包含 `_thumb` 的缩略图
  - 跳过 `Thumbs.db` 等系统文件
  - 跳过文件大小小于 3KB 的图片
  - **不处理 .webp 格式**
  - **同名不同后缀时，优先选择 .png > .jpeg > .jpg**（其他后缀版本被忽略）
- **图片预处理（可选）**: 使用 ffmpeg 将图片缩放到 672x672 (保持宽高比)，可通过 `FASHIONCLIP_RESIZE=false` 禁用
- 生成 512 维向量 (CLIP ViT-B/16)
- 支持增量更新 (只处理新增/修改的图片)

### 2. 相似图片搜索

- 上传图片或提供图片 URL
- 返回 top-k 最相似图片
- 支持相似度阈值过滤
- 返回结果包含:
  - 相似度分数
  - 本地路径 (`/mnt/dapai-s/...`)
  - Rclone HTTP 路径 (`http://192.168.0.10:8080/...`)
  - 图片基本信息 (大小、格式)

### 3. 任务计划

- 支持按目录扫描并批量 embedding
- 记录处理进度到 Redis
- 支持增量更新 (只处理新增/修改的图片)

## API 接口

### 基础信息

- **Base URL**: `http://localhost:8008`
- **端口**: 8008

### 1. 健康检查

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "photos_dir": "/mnt/dapai-s"
}
```

### 2. 搜索相似图片

```
POST /search
Content-Type: multipart/form-data
```

**Request Body (form-data)**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 否 | 上传的图片文件 (JPEG/PNG/WebP) |
| image_url | string | 否 | 图片 URL |
| top_k | int | 否 | 返回数量，默认 10 |
| threshold | float | 否 | 相似度阈值，默认 0.0 |

> `file` 和 `image_url` 二选一

**Response**:

```json
{
  "query_type": "upload",
  "results": [
    {
      "path": "/mnt/dapai-s/category/dress/001.jpg",
      "rclone_url": "http://192.168.0.10:8080/category/dress/001.jpg",
      "score": 0.9523,
      "size": 204800,
      "format": "jpeg"
    }
  ],
  "total": 1,
  "query_time_ms": 156
}
```

### 3. 触发目录扫描 (异步)

```
POST /embed/scan
```

**Request Body**:

```json
{
  "force_refresh": false
}
```

- `force_refresh`: false = 增量 (只处理新文件)，true = 全量重新处理

**Response**:

```json
{
  "task_id": "scan_20260328_143052",
  "status": "started",
  "message": "目录扫描任务已启动",
  "estimated_count": 1523
}
```

### 4. 批量 Embedding

```
POST /embed/batch
Content-Type: multipart/form-data
```

**Request Body (form-data)**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | List[File] | 是 | 图片文件列表 (最多 20 个) |

**Response**:

```json
{
  "count": 5,
  "success": true,
  "errors": []
}
```

### 5. 获取扫描状态

```
GET /embed/status
```

**Response**:

```json
{
  "is_scanning": true,
  "progress": {
    "total": 1523,
    "processed": 456,
    "failed": 3,
    "percentage": 29.9
  },
  "last_scan": "2026-03-28T14:30:52Z",
  "total_indexed": 1520
}
```

### 6. 获取单张图片向量

```
GET /embed/{image_path}
```

**Path Parameters**:
- `image_path`: URL 编码的图片相对路径，如 `category/dress/001.jpg`

**Response**:

```json
{
  "path": "/mnt/dapai-s/category/dress/001.jpg",
  "vector_dim": 512,
  "indexed_at": "2026-03-28T14:30:52Z"
}
```

## 配置环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PHOTOS` | `/mnt/dapai-s` | 服装图片根目录 |
| `RCLONE_BASE_URL` | `http://192.168.0.10:8080` | Rclone HTTP 服务地址 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 服务地址 |
| `QDRANT_COLLECTION` | `images` | Qdrant 集合名 (与 similar-photo 共用) |
| `QDRANT_API_KEY` | - | Qdrant API Key (可选) |
| `REDIS_URL` | `redis://localhost:6379` | Redis 地址 (用于任务状态) |
| `BATCH_SIZE` | `32` | 批处理大小 |
| `DEVICE` | `cuda` | 运行设备 (cuda/cpu) |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用 ffmpeg 图片压缩 |
| `FASHIONCLIP_MAX_DIM` | `672` | 压缩后最大尺寸 (px) |
| `FASHIONCLIP_QUALITY` | `q85` | JPEG 压缩质量 |

## 目录结构

```
/mnt/dapai-s/
├── category1/
│   ├── item_001.jpg
│   ├── item_001.jpeg
│   └── item_001.png    # 同名文件只选择 .png
├── category2/
│   └── item_002.webp   # .webp 不处理
└── ...
```

## 向量格式

- **维度**: 512 (CLIP ViT-B/16)
- **存储**: Qdrant (collection: `images`，与 chujiang_alioss_similar 共用同一空间)
- **Payload**:
  ```json
  {
    "path": "/mnt/dapai-s/category/item_001.jpg",
    "rclone_url": "http://192.168.0.10:8080/category/item_001.jpg",
    "size": 204800,
    "format": "jpeg",
    "indexed_at": "2026-03-28T14:30:52Z"
  }
  ```

## 与阿里百炼 API 的兼容接口

为方便替换 `aliOpenaiClient`，提供兼容层：

```typescript
// 阿里云百炼格式 (aliOpenai.ts)
interface AliEmbeddingOutput {
  embeddings: Array<{
    embedding: number[];
    index: number;
    type: string;
  }>;
}

// FashionCLIP API 响应格式
interface FashionClipResponse {
  embedding: number[];
  model: "fashionclip";
}
```

转换适配器可在 `src/utils/fashionclip-adapter.ts` 中实现。

## 部署方式

### Docker

```bash
# 构建
docker build -t fashionclip-api .

# 运行
docker run -d \
  --gpus all \
  -p 8008:8008 \
  -v /mnt/dapai-s:/mnt/dapai-s:ro \
  -e PHOTOS=/mnt/dapai-s \
  -e RCLONE_BASE_URL=http://192.168.0.10:8080 \
  -e QDRANT_URL=http://qdrant:6333 \
  --name fashionclip-api \
  fashionclip-api
```

### Docker Compose (参考 similar-photo 项目)

```yaml
services:
  fashionclip-api:
    build: .
    ports:
      - "8008:8008"
    volumes:
      - /mnt/dapai-s:/mnt/dapai-s:ro
    environment:
      - PHOTOS=/mnt/dapai-s
      - RCLONE_BASE_URL=http://192.168.0.10:8080
      - QDRANT_URL=http://qdrant:6333
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 注意事项

1. **首次启动**: FashionCLIP 模型约 900MB，首次调用时下载并缓存
2. **GPU 推荐**: 无 GPU 时推理较慢 (单图约 500ms CPU vs 50ms GPU)
3. **向量空间**: 与阿里百炼的向量**不在同一空间**，如需切换必须重新 embedding 全量图片
4. **Rclone**: 图片目录需通过 Rclone HTTP 暴露，或修改 `RCLONE_BASE_URL` 为实际地址
