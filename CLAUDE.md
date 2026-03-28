# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FashionCLIP 图片相似度搜索 API 服务 - 基于 FashionCLIP 模型 (laion/CLIP-ViT-B-16-laion2B-s34B-b88K) 的本地图片向量化和相似图片搜索服务，专为服饰鞋帽行业设计。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python app.py

# Docker 构建
docker build -t fashionclip-api .

# Docker 运行 (需要 GPU)
docker run -d --gpus all \
  -p 8008:8008 \
  -v /mnt/dapai-s:/mnt/dapai-s:ro \
  -e PHOTOS=/mnt/dapai-s \
  -e RCLONE_BASE_URL=http://192.168.0.10:8080 \
  -e QDRANT_URL=http://localhost:6333 \
  --name fashionclip-api \
  fashionclip-api
```

## 技术栈

- **运行时**: Python 3.10 + CUDA 12.1
- **框架**: FastAPI
- **模型**: FashionCLIP / CLIP ViT-B/16 (512 维向量)
- **向量数据库**: Qdrant (复用 similar-photo 项目的实例)
- **镜像**: nvidia/cuda:12.1.0-runtime-ubuntu22.04

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/search` | 搜索相似图片 (上传文件或 URL) |
| POST | `/embed/scan` | 触发目录扫描 (异步) |
| POST | `/embed/batch` | 批量 embedding |
| GET | `/embed/status` | 获取扫描状态 |
| GET | `/embed/{image_path}` | 获取单张图片向量信息 |

## 配置环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PHOTOS` | `/mnt/dapai-s` | 服装图片根目录 |
| `RCLONE_BASE_URL` | `http://192.168.0.10:8080` | Rclone HTTP 服务 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 地址 (复用 similar-photo) |
| `QDRANT_COLLECTION` | `images` | Qdrant 集合名 (与 chujiang_alioss_similar 共用) |
| `QDRANT_API_KEY` | - | Qdrant API Key (可选) |
| `DEVICE` | `cuda` | 运行设备 |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用 ffmpeg 图片压缩 |
| `FASHIONCLIP_MAX_DIM` | `672` | 压缩后最大尺寸 |
| `FASHIONCLIP_QUALITY` | `q85` | JPEG 压缩质量 |

## 与 chujiang_alioss_similar 的关系

- **向量空间统一**: 使用相同的 Qdrant collection (`images`)，向量可直接对比
- **chujiang_alioss_similar 改造**: 后续将移除阿里百炼调用，改为请求此服务的 `/search` 接口获取 embedding
- **部署位置**: 建议与 chujiang_alioss_similar 部署在同一机器或同一 Docker 网络中

## 注意事项

1. 首次启动时 FashionCLIP 模型 (约 900MB) 会自动下载到 `/code/cache`
2. 必须使用 GPU (`--gpus all`) 以保证推理性能
3. `/mnt/dapai-s` 目录通过只读挂载 (`ro`) 防止意外修改
4. 图片预处理 (ffmpeg) 可通过 `FASHIONCLIP_RESIZE=false` 禁用
