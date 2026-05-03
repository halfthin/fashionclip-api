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
            model_name="ViT-B-16",
            pretrained="laion2B-s34B-b88K",
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
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FashionCLIP Embedding API 启动中...")
    load_fashionclip_model()
    logger.info("服务就绪")
    yield


app = FastAPI(title="FashionCLIP Embedding API", lifespan=lifespan)

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

    if not req.paths:
        return {"embeddings": [], "errors": []}

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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
