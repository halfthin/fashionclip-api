# FashionCLIP API 精简实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 fashionclip-api 从包含 Qdrant 数据库的全功能服务精简为纯无状态的 CLIP embedding 服务，只暴露 3 个端点匹配 ht-files 调用需求。

**Architecture:** 保留 open_clip 模型加载 + 图片预处理逻辑，移除所有数据库操作（Qdrant）、文件扫描、搜索功能。输入从 HTTP URL 改为 `{base, path(s)}` 本地路径模式（同时兼容 HTTP URL）。

**Tech Stack:** Python 3.10, FastAPI, open_clip, PyTorch, Pillow

---

### Task 1: Git 重置到可工作基线

**Files:** 工作树重置

- [ ] **Step 1: 丢弃所有未提交改动**

Run: `git -C /home/halfthin/dev/sop/fashionclip-api reset --hard`

Expected: 所有未提交改动被丢弃

- [ ] **Step 2: 重置到 eaadb90 提交**

Run: `git -C /home/halfthin/dev/sop/fashionclip-api reset --hard eaadb90`

Expected: 回到 Qdrant 可工作的状态，app.py 包含 open_clip + 完整功能

- [ ] **Step 3: 验证重置后状态**

```bash
cd /home/halfthin/dev/sop/fashionclip-api && git log --oneline -3
# 应该显示:
# eaadb90 chore: 更新 .gitignore 排除 Claude Code 和本地模型目录
# 6c54dbf feat: 添加 OpenCLIP 到 HuggingFace CLIP 模型转换脚本
# 2159d7d feat: 支持扫描 wxwork-media 共享目录
```

Then ensure the spec doc commit is still reachable (cherry-pick if needed):

```bash
git cherry-pick af5e1e6  # 恢复设计文档提交
```

---

### Task 2: 重写 app.py

**Files:**
- Modify: `app.py` — 从 ~950 行精简到 ~200 行

- [ ] **Step 1: 用以下内容完全替换 app.py**

```python
"""
FashionCLIP Embedding API
纯无状态 CLIP 图片向量化服务，不连接任何数据库。
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from urllib.request import urlopen

import torch
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import open_clip

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashionclip")

# ============ 配置 ============
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ENABLE_IMAGE_RESIZE = os.getenv("FASHIONCLIP_RESIZE", "true").lower() == "true"
IMAGE_MAX_DIMENSION = int(os.getenv("FASHIONCLIP_MAX_DIM", "672"))
IMAGE_QUALITY = int(os.getenv("FASHIONCLIP_QUALITY", "85"))

# ============ 全局模型 ============
model = None
preprocess = None


def load_fashionclip_model():
    """加载 FashionCLIP 模型（OpenCLIP 格式）"""
    global model, preprocess
    if model is not None:
        return model, preprocess

    logger.info("正在加载 FashionCLIP 模型...")
    cache_dir = "/code/cache"
    model_base = Path(cache_dir) / "models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K" / "snapshots"

    safetensors_path = None
    if model_base.exists():
        for snap_dir in model_base.iterdir():
            if snap_dir.is_dir():
                candidate = snap_dir / "open_clip_model.safetensors"
                if candidate.exists():
                    safetensors_path = candidate
                    break

    if safetensors_path:
        logger.info(f"从本地缓存加载: {safetensors_path}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained=str(safetensors_path),
        )
    else:
        logger.info("从 HuggingFace 加载模型...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        )

    model = model.to(DEVICE)
    model.eval()
    logger.info(f"模型加载完成，设备: {DEVICE}")
    return model, preprocess


def resize_image(img: Image.Image) -> Image.Image:
    """用 PIL 缩放图片，保持宽高比"""
    if not ENABLE_IMAGE_RESIZE:
        return img
    w, h = img.size
    max_dim = IMAGE_MAX_DIMENSION
    if w <= max_dim and h <= max_dim:
        return img
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def read_image(source: str, base: str) -> Image.Image:
    """从本地文件系统或 HTTP URL 读取图片。

    参数:
        base:  非空时作为 /mnt/{base} 路径前缀，空字符串表示 HTTP URL 模式
        source: base 非空时为相对路径，base 为空时为完整 HTTP URL
    """
    if base:
        if "/" in base or base in (".", ".."):
            raise ValueError("base 不合法")
        if ".." in source:
            raise ValueError("path 不合法")
        full_path = f"/mnt/{base}/{source.lstrip('/')}"
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"文件不存在: {full_path}")
        raw = Image.open(full_path).convert("RGB")
    else:
        with urlopen(source, timeout=30) as resp:
            raw = Image.open(io.BytesIO(resp.read())).convert("RGB")
    img = resize_image(raw)
    if img is not raw:
        raw.close()
    return img


def get_embedding(image: Image.Image) -> List[float]:
    """单张图片 CLIP 编码 + L2 归一化"""
    m, preproc = load_fashionclip_model()
    with torch.no_grad():
        tensor = preproc(image).unsqueeze(0).to(DEVICE)
        features = m.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


def get_embeddings_batch(images: List[Image.Image]) -> List[List[float]]:
    """批量 CLIP 编码 + L2 归一化（GPU 友好）"""
    if not images:
        return []
    m, preproc = load_fashionclip_model()
    with torch.no_grad():
        tensors = torch.stack([preproc(img) for img in images]).to(DEVICE)
        features = m.encode_image(tensors)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()


# ============ FastAPI 应用 ============
app = FastAPI(title="FashionCLIP Embedding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    base: str = Field(..., description="挂载目录名（如 dapai-s），空字符串表示 HTTP URL 模式")
    path: str = Field(..., description="相对路径或 HTTP URL")


class EmbedBatchRequest(BaseModel):
    base: str = Field(..., description="挂载目录名，空字符串表示 HTTP URL 模式")
    paths: List[str] = Field(..., max_length=20, description="路径列表，最多 20 个")


# ============ 路由 ============
@app.post("/embed")
async def embed(req: EmbedRequest):
    """单图向量化。base 非空时从本地读取，base 为空时从 HTTP URL 下载。"""
    try:
        img = read_image(req.path, req.base)
        vector = get_embedding(img)
        img.close()
        return {"embedding": vector}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片处理失败: {str(e)}")


@app.post("/embed-batch")
async def embed_batch(req: EmbedBatchRequest):
    """批量向量化。并行读取图片后一次 GPU 前向推理。"""
    load_fashionclip_model()

    # 并行加载
    loaded: dict[str, Image.Image | None] = {}
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(8, len(req.paths))) as ex:
        fut_map = {ex.submit(read_image, p, req.base): p for p in req.paths}
        for f in as_completed(fut_map):
            p = fut_map[f]
            try:
                loaded[p] = f.result()
            except Exception as e:
                errors.append({"path": p, "error": str(e)})
                loaded[p] = None

    valid = [(p, loaded[p]) for p in req.paths if loaded.get(p) is not None]
    if not valid:
        return {"embeddings": [], "errors": errors}

    # 批量推理（保持原始顺序）
    images = [img for _, img in valid]
    try:
        vectors = get_embeddings_batch(images)
    except Exception as e:
        for _, img in valid:
            img.close()
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    for _, img in valid:
        img.close()

    return {"embeddings": vectors, "errors": errors}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE,
    }


@app.on_event("startup")
async def startup():
    """启动时加载模型"""
    logger.info("FashionCLIP Embedding API 启动中...")
    load_fashionclip_model()
    logger.info("服务就绪")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
```

- [ ] **Step 2: 验证 app.py 能通过语法检查**

```bash
cd /home/halfthin/dev/sop/fashionclip-api && python3 -c "import ast; ast.parse(open('app.py').read()); print('语法 OK')"
```

Expected: `语法 OK`

---

### Task 3: 更新 requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 替换为精简后的依赖**

```text
open_clip_torch
fastapi==0.110.0
uvicorn[standard]==0.27.0
torch==2.1.0
pillow==10.2.0
numpy==1.26.4
python-dotenv==1.0.1
python-multipart==0.0.9
```

**删除**: `psycopg2-binary`, `pgvector`, `scikit-learn`, `qdrant-client`

---

### Task 4: 更新 Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: 替换为精简后的 Dockerfile**

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    gosu \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

COPY requirements.txt .
RUN pip install --no-cache-dir --no-warn-script-location -r requirements.txt && \
    HOME=/root pip install --no-cache-dir --no-warn-script-location open-clip-torch

COPY --chown=appuser:appuser app.py /code/app.py
COPY --chown=appuser:appuser entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh

RUN mkdir -p /code/cache /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /code /home/appuser/.cache

ENV DEVICE=cuda
ENV FASHIONCLIP_RESIZE=true
ENV FASHIONCLIP_MAX_DIM=672
ENV FASHIONCLIP_QUALITY=85
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

ENTRYPOINT ["/code/entrypoint.sh"]
```

变化说明：
- 删除了 `PHOTOS`, `IMAGE_BASE_URL`, `ALIST_BASE_URL`, `QDRANT_*`, `BATCH_SIZE` 环境变量
- 只保留 DEVICE + FASHIONCLIP_* 相关变量
- GPU 不再是强制要求（但镜像本身仍是 CUDA base）

---

### Task 5: 更新 docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: 替换为精简后的 docker-compose.yml**

```yaml
services:
  fashionclip-api:
    image: halfthin/fashion:latest
    container_name: fashionclip-api
    restart: always
    env_file: .env.docker
    ports:
      - "${HOST_PORT:-8008}:8008"
    volumes:
      - /mnt/dapai-s:/mnt/dapai-s:ro
      - /mnt/wxwork-media:/mnt/wxwork-media:ro
      - ./data/models:/code/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8008/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - web_net

networks:
  web_net:
    external: true
```

变化说明：
- 删除了网络配置（`web_net` 如果没有特别需要可以去掉，保留兼容现网配置）
- 移除了 Qdrant 依赖相关的注释
- 其他基本保持不变，GPU reservation 保留但同理 CPU 模式可关

---

### Task 6: 更新环境变量文件

**Files:**
- Modify: `.env.example`
- Modify: `.env.docker`

- [ ] **Step 1: 替换 .env.example**

```bash
# ============ 向量化配置 ============
# 运行设备 (cuda/cpu)
DEVICE=cuda

# ============ 图片压缩 (可选) ============
# 是否启用 ffmpeg 图片缩放 (true/false)
FASHIONCLIP_RESIZE=true
# 缩放后最大尺寸 (px)
FASHIONCLIP_MAX_DIM=672
# JPEG 压缩质量 (1-31，越小质量越低)
FASHIONCLIP_QUALITY=85

# HuggingFace 镜像
# HF_ENDPOINT=https://hf-mirror.com
```

- [ ] **Step 2: 替换 .env.docker**

```bash
DEVICE=cuda
FASHIONCLIP_RESIZE=true
FASHIONCLIP_MAX_DIM=672
FASHIONCLIP_QUALITY=85
HF_ENDPOINT=https://hf-mirror.com
HOST_PORT=8008
```

---

### Task 7: 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 替换为精简后的内容**

```markdown
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
```

---

### Task 8: 重写 API.md

**Files:**
- Modify: `API.md`

- [ ] **Step 1: 替换为精简后的 API 文档**

```markdown
# FashionCLIP Embedding API 文档

纯无状态 CLIP 图片向量化服务，不连接任何数据库。

**Swagger UI**: `http://<host>:8008/docs`

## 基础信息

- **服务地址**: `http://<host>:8008`
- **向量维度**: 512 维 (CLIP ViT-B/16)
- **模型**: FashionCLIP `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`
- **状态**: 无状态

## API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/embed` | 单图向量化 |
| POST | `/embed-batch` | 批量向量化（≤20 张） |
| GET | `/health` | 健康检查 |

---

## 1. 健康检查

### 请求

```
GET /health
```

### 响应

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 健康状态: `ok` |
| `model_loaded` | Boolean | 模型是否已加载 |
| `device` | String | 运行设备: `cuda` 或 `cpu` |

---

## 2. 单图向量化

### 请求

```
POST /embed
Content-Type: application/json
```

### 参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `base` | String | 是 | 挂载目录名（如 `dapai-s`），空字符串表示 HTTP URL 模式 |
| `path` | String | 是 | 相对路径或 HTTP URL |

### 路径模式

- **本地模式** (`base` 非空): 拼接为 `/mnt/{base}/{path}`
- **URL 模式** (`base` 为空): `path` 作为 HTTP URL 直接下载

### 示例: 本地文件

```bash
curl -X POST http://localhost:8008/embed \
  -H "Content-Type: application/json" \
  -d '{"base": "dapai-s", "path": "2026年/3月/0331/img.jpg"}'
```

### 示例: HTTP URL

```bash
curl -X POST http://localhost:8008/embed \
  -H "Content-Type: application/json" \
  -d '{"base": "", "path": "http://100.64.0.6:5244/dapai-s/img.jpg"}'
```

### 响应

```json
{
  "embedding": [0.12, -0.34, 0.56, ...]
}
```

### 错误响应

```json
{
  "detail": "文件不存在: /mnt/dapai-s/path/to/img.jpg"
}
```

---

## 3. 批量向量化

### 请求

```
POST /embed-batch
Content-Type: application/json
```

### 参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `base` | String | 是 | 挂载目录名，空字符串表示 HTTP URL 模式 |
| `paths` | String[] | 是 | 路径列表，最多 20 个 |

### 示例: 本地批量

```bash
curl -X POST http://localhost:8008/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"base": "dapai-s", "paths": ["img1.jpg", "img2.jpg"]}'
```

### 示例: URL 批量

```bash
curl -X POST http://localhost:8008/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"base": "", "paths": ["http://example.com/1.jpg", "http://example.com/2.jpg"]}'
```

### 响应

```json
{
  "embeddings": [
    [0.12, -0.34, ...],
    [0.56, 0.78, ...]
  ],
  "errors": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `embeddings` | Number[][] | 向量列表，与输入路径顺序一致 |
| `errors` | Object[] | 处理失败的条目（失败时不整体报错） |

---

## 错误处理

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 400 | 参数错误（base 不合法、路径不存在、图片无法解析） |
| 500 | 服务器内部错误（推理失败） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` | 运行设备 |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用图片缩放 |
| `FASHIONCLIP_MAX_DIM` | `672` | 缩放后最大尺寸 |
| `FASHIONCLIP_QUALITY` | `85` | JPEG 压缩质量 |
```

---

### Task 9: 重写 SPEC.md

**Files:**
- Modify: `SPEC.md`

- [ ] **Step 1: 替换为精简后的规格**

```markdown
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
```

---

### Task 10: 最终提交

**Files:** 所有修改过的文件

- [ ] **Step 1: 检查所有未提交文件**

```bash
cd /home/halfthin/dev/sop/fashionclip-api && git status
```

Expected: 只有计划中的文件有变动

- [ ] **Step 2: 提交所有变更**

```bash
git add app.py requirements.txt Dockerfile docker-compose.yml .env.example .env.docker \
       CLAUDE.md API.md SPEC.md docs/superpowers/specs/2026-05-03-fashionclip-simplify-design.md && \
git commit -m "$(cat <<'EOF'
refactor: 精简为纯无状态 CLIP embedding 服务

- 移除 Qdrant 数据库集成和所有数据库操作
- 移除 /search, /embed/scan, /embed/status, /embed/cancel, /embed/{path} 端点
- 移除文件扫描引擎和 checkpoint 机制
- 新增 /embed 和 /embed-batch 端点，支持 {base, path(s)} 本地路径模式
- 同时兼容 HTTP URL 模式（base 为空时）
- 精简环境变量和依赖
- 更新所有文档

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: 验证提交成功**

```bash
cd /home/halfthin/dev/sop/fashionclip-api && git log --oneline -3
```

Expected: 看到最新的提交消息
