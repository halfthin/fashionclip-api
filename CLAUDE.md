# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FashionCLIP Embedding API - 基于 FashionCLIP 模型 (laion/CLIP-ViT-B-16-laion2B-s34B-b88K) 的轻量图片向量化服务。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python app.py

# Docker 构建
docker build -t fashionclip-api .

# Docker 运行 (GPU)
docker run -d --gpus all \
  -p 8008:8008 \
  -v /mnt/dapai-s:/mnt/dapai-s:ro \
  --name fashionclip-api \
  fashionclip-api

# Docker 运行 (CPU)
docker run -d -p 8008:8008 -e DEVICE=cpu fashionclip-api
```

## 技术栈

- **运行时**: Python 3.10 + CUDA 12.1 (GPU 可选)
- **框架**: FastAPI
- **模型**: FashionCLIP / CLIP ViT-B/16 (512 维向量)
- **状态**: 纯无状态服务，不连接任何数据库

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/embed` | 单图向量化 — `{base, path}` → `{embedding}` |
| POST | `/embed-batch` | 批量向量化 — `{base, paths}` → `{embeddings}` |
| GET | `/health` | 健康检查 |

## 配置环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` | 运行设备 (cuda/cpu) |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用 ffmpeg 图片缩放 |
| `FASHIONCLIP_MAX_DIM` | `672` | 缩放后最大尺寸 |
| `FASHIONCLIP_QUALITY` | `85` | JPEG 压缩质量 |

## 注意事项

1. 首次启动时 FashionCLIP 模型 (约 900MB) 会自动下载到 `/code/cache`
2. GPU 模式需 `--gpus all`，CPU 模式设 `DEVICE=cpu`
3. 图片目录通过 NFS 挂载，`base` 映射到 `/mnt/{base}` 目录
4. `base` 为空时，`path` 作为 HTTP URL 处理
