# FashionCLIP Embedding API

## 概述

基于 FashionCLIP 模型的轻量无状态图片向量化服务。

## 技术栈

- **运行时**: Python 3.10 + CUDA 12.1 (GPU 可选)
- **框架**: FastAPI
- **模型**: FashionCLIP (laion/CLIP-ViT-B-16-laion2B-s34B-b88K)
- **状态**: 纯无状态，不连接任何数据库

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/embed` | 单图向量化 |
| POST | `/embed-batch` | 批量向量化（≤20） |

### POST /embed

请求: `{"base": "dapai-s", "path": "relative/path.jpg"}`
响应: `{"embedding": [0.12, -0.34, ...]}`

`base` 映射为 `/mnt/{base}`，`base` 为空时 `path` 作为 HTTP URL。

### POST /embed-batch

请求: `{"base": "dapai-s", "paths": ["a.jpg", "b.jpg"]}`
响应: `{"embeddings": [[...], [...]], "errors": []}`

### GET /health

响应: `{"status": "ok", "model_loaded": true, "device": "cuda"}`

## 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` | 运行设备 (cuda/cpu) |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用图片缩放 |
| `FASHIONCLIP_MAX_DIM` | `672` | 缩放后最大尺寸 |
| `FASHIONCLIP_QUALITY` | `85` | JPEG 压缩质量 |

## 部署

### GPU
```bash
docker run -d --gpus all -p 8008:8008 -v /mnt/dapai-s:/mnt/dapai-s:ro fashionclip-api
```

### CPU
```bash
docker run -d -p 8008:8008 -e DEVICE=cpu fashionclip-api
```
