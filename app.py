"""
FashionCLIP 图片相似度搜索 API 服务
基于 FashionCLIP 模型，为服饰鞋帽行业提供本地化的图像向量化和搜索服务。
"""

import hashlib
import io
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import CLIPProcessor, CLIPModel

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashionclip")

# ============ 配置 ============
PHOTOS_DIR = os.getenv("PHOTOS", "/mnt/dapai-s")
RCLONE_BASE_URL = os.getenv("RCLONE_BASE_URL", "http://192.168.0.10:8080")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "images")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 图片过滤规则
SKIP_PREFIXES = ("._", "Thumbs.db", ".DS_Store")
SKIP_SUBSTRINGS = ("_thumb", "_thumbnail", "_small")
MIN_FILE_SIZE = 3 * 1024  # 3KB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # 不处理 .webp
# 同名不同后缀时的优先级
EXT_PRIORITY = {".png": 1, ".jpeg": 2, ".jpg": 3}

# ============ 图片压缩配置 (可选) ============
# 使用 ffmpeg 预处理图片，可降低 FashionCLIP 处理负担
ENABLE_IMAGE_RESIZE = os.getenv("FASHIONCLIP_RESIZE", "true").lower() == "true"
IMAGE_MAX_DIMENSION = int(os.getenv("FASHIONCLIP_MAX_DIM", "672"))  # 672x672 (CLIP 常用输入)
IMAGE_QUALITY = int(os.getenv("FASHIONCLIP_QUALITY", "q85"))  # JPEG 质量

# ============ 全局状态 ============
model = None
processor = None
qdrant_client = None
scan_status = {
    "is_scanning": False,
    "progress": {"total": 0, "processed": 0, "failed": 0},
    "last_scan": None,
    "total_indexed": 0,
}


# ============ FastAPI 应用 ============
app = FastAPI(
    title="FashionCLIP 图片相似度搜索 API",
    description="基于 FashionCLIP 的本地图片向量化和相似图片搜索服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 工具函数 ============
def load_fashionclip_model():
    """加载 FashionCLIP 模型"""
    global model, processor
    if model is None:
        logger.info("正在加载 FashionCLIP 模型...")
        model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            cache_dir="/code/cache",
        )
        processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            cache_dir="/code/cache",
        )
        model = model.to(DEVICE)
        model.eval()
        logger.info(f"FashionCLIP 模型加载完成，设备: {DEVICE}")
    return model, processor


def init_qdrant():
    """初始化 Qdrant 客户端和集合"""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10,
        )
        try:
            qdrant_client.get_collection(QDRANT_COLLECTION)
            logger.info(f"Qdrant 集合已存在: {QDRANT_COLLECTION}")
        except (UnexpectedResponse, Exception):
            logger.info(f"创建 Qdrant 集合: {QDRANT_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "image": qdrant_models.VectorParams(
                        size=512,
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
            )
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="path",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )


def is_valid_image(filename: str, file_size: int) -> bool:
    """检查图片是否有效"""
    # 检查扩展名
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # 跳过特定前缀
    for prefix in SKIP_PREFIXES:
        if filename.startswith(prefix):
            return False

    # 跳过包含特定字符串的文件名
    for substring in SKIP_SUBSTRINGS:
        if substring.lower() in filename.lower():
            return False

    # 跳过太小文件
    if file_size < MIN_FILE_SIZE:
        return False

    return True


def preprocess_image(input_path: str) -> Image.Image:
    """
    图片预处理：使用 ffmpeg 压缩/缩放图片 (文件路径版)

    可选功能，通过 FASHIONCLIP_RESIZE=true/false 开关
    将图片缩放到 IMAGE_MAX_DIMENSION 尺寸，保持宽高比
    """
    if not ENABLE_IMAGE_RESIZE:
        return Image.open(input_path).convert("RGB")

    import subprocess

    # 构建 ffmpeg 命令
    # scale=iw:ih -> 保持宽高比，max(ih,iw)<=MAX_DIMENSION
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale='min(iw\\,{IMAGE_MAX_DIMENSION}):min(ih\\,{IMAGE_MAX_DIMENSION})'",
        "-q:v", str(IMAGE_QUALITY),
        "-f", "image2pipe",
        "-vcodec", "png",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffmpeg 压缩失败，fallback 到原图: {input_path}")
            return Image.open(input_path).convert("RGB")

        from io import BytesIO
        return Image.open(BytesIO(result.stdout)).convert("RGB")
    except Exception as e:
        logger.warning(f"图片预处理失败，fallback 到原图: {e}")
        return Image.open(input_path).convert("RGB")


def resize_image_pil(img: Image.Image) -> Image.Image:
    """
    使用 PIL 缩放图片 (内存版，用于上传/URL 图片)
    保持宽高比，缩放到 IMAGE_MAX_DIMENSION
    """
    if not ENABLE_IMAGE_RESIZE:
        return img

    # 计算缩放后的尺寸
    w, h = img.size
    max_dim = IMAGE_MAX_DIMENSION

    if w <= max_dim and h <= max_dim:
        return img  # 已经足够小

    # 按比例缩放
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def get_image_embedding(image: Image.Image) -> List[float]:
    """获取单张图片的 embedding 向量"""
    m, p = load_fashionclip_model()
    inputs = p(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = m.get_image_features(**inputs)
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()


def image_to_rclone_url(local_path: str) -> str:
    """将本地路径转换为 Rclone HTTP URL"""
    # /mnt/dapai-s/category/item.jpg -> /category/item.jpg
    if local_path.startswith(PHOTOS_DIR):
        relative_path = local_path[len(PHOTOS_DIR):].lstrip("/")
    else:
        relative_path = local_path.lstrip("/")
    return f"{RCLONE_BASE_URL.rstrip('/')}/{relative_path}"


def scan_photos_directory() -> List[dict]:
    """扫描照片目录，返回所有有效图片信息

    过滤规则：
    - 跳过 ._ 开头、Thumbs.db 等系统文件
    - 跳过包含 _thumb 等的缩略图
    - 跳过小于 3KB 的文件
    - 不处理 .webp 格式
    - 同名不同后缀时，优先选择 .png > .jpeg > .jpg
    """
    photos_path = Path(PHOTOS_DIR)
    if not photos_path.exists():
        raise ValueError(f"照片目录不存在: {PHOTOS_DIR}")

    # 第一步：收集所有有效图片，按 base name 分组
    name_to_files: dict[str, list[dict]] = {}
    for root, dirs, files in os.walk(photos_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                file_size = os.path.getsize(filepath)
                if not is_valid_image(filename, file_size):
                    continue

                ext = Path(filename).suffix.lower()
                base_name = str(Path(filename).with_suffix(""))
                relative_path = filepath[len(str(photos_path)) + 1:]  # 相对于 photos_dir

                if base_name not in name_to_files:
                    name_to_files[base_name] = []

                name_to_files[base_name].append({
                    "path": filepath,
                    "size": file_size,
                    "format": ext.replace(".", ""),
                    "relative_path": relative_path,
                })
            except OSError:
                continue

    # 第二步：每个 base name 只选一个文件（按优先级）
    image_files = []
    for base_name, files in name_to_files.items():
        if len(files) == 1:
            # 只有一个文件，直接使用
            image_files.append(files[0])
        else:
            # 有多个同名不同后缀的文件，按优先级选择
            files.sort(key=lambda x: EXT_PRIORITY.get(f".{x['format']}", 99))
            chosen = files[0]
            image_files.append(chosen)
            # 记录跳过的文件（调试用）
            skipped = files[1:]
            if skipped:
                logger.debug(
                    f"同名文件选择: {chosen['relative_path']}, 跳过: {[f['relative_path'] for f in skipped]}"
                )

    return image_files


# ============ API 路由 ============
@app.get("/health")
async def health_check():
    """健康检查"""
    model_loaded = model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "photos_dir": PHOTOS_DIR,
        "qdrant_url": QDRANT_URL,
    }


class SearchRequest(BaseModel):
    image_url: Optional[str] = None
    top_k: int = 10
    threshold: float = 0.0


@app.post("/search")
async def search_similar(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    top_k: int = Form(10),
    threshold: float = Form(0.0),
):
    """
    搜索相似图片
    - file: 上传的图片文件 (JPEG/PNG/WebP)
    - image_url: 图片 URL (与 file 二选一)
    - top_k: 返回数量
    - threshold: 相似度阈值
    """
    start_time = time.time()

    # 加载模型 (如果尚未加载)
    load_fashionclip_model()
    init_qdrant()

    # 获取查询图片
    query_image = None
    query_type = "unknown"

    if file:
        query_type = "upload"
        try:
            contents = await file.read()
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
            query_image = resize_image_pil(raw_image)
            if query_image is not raw_image:
                raw_image.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析图片: {str(e)}")
    elif image_url:
        query_type = "url"
        try:
            import urllib.request
            with urllib.request.urlopen(image_url, timeout=10) as response:
                contents = response.read()
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
            query_image = resize_image_pil(raw_image)
            if query_image is not raw_image:
                raw_image.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法下载图片: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="必须提供 file 或 image_url")

    # 获取查询向量
    query_vector = get_image_embedding(query_image)
    query_image.close()

    # 搜索 Qdrant
    try:
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=("image", query_vector),
            limit=top_k,
            score_threshold=threshold,
        )
    except Exception as e:
        logger.error(f"Qdrant 搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

    # 格式化结果
    search_results = []
    for hit in results:
        payload = hit.payload or {}
        search_results.append({
            "path": payload.get("path", ""),
            "rclone_url": payload.get("rclone_url", ""),
            "score": round(hit.score, 4),
            "size": payload.get("size", 0),
            "format": payload.get("format", ""),
        })

    query_time_ms = round((time.time() - start_time) * 1000)

    return {
        "query_type": query_type,
        "results": search_results,
        "total": len(search_results),
        "query_time_ms": query_time_ms,
    }


@app.post("/embed/scan")
async def trigger_scan(force_refresh: bool = Form(False)):
    """
    触发目录扫描 (异步)
    - force_refresh: false = 增量, true = 全量重新处理
    """
    global scan_status

    if scan_status["is_scanning"]:
        return {
            "task_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "already_running",
            "message": "扫描任务正在进行中",
        }

    # 启动后台扫描
    import asyncio

    async def run_scan():
        global scan_status
        scan_status["is_scanning"] = True
        scan_status["progress"] = {"total": 0, "processed": 0, "failed": 0}

        try:
            # 初始化 Qdrant
            init_qdrant()

            # 扫描目录
            image_files = scan_photos_directory()
            total = len(image_files)
            scan_status["progress"]["total"] = total
            scan_status["progress"]["processed"] = 0
            scan_status["progress"]["failed"] = 0

            logger.info(f"开始扫描 {total} 张图片...")

            # 批量处理
            m, p = load_fashionclip_model()
            points = []

            for i, img_info in enumerate(image_files):
                try:
                    # 检查是否已索引 (增量模式)
                    if not force_refresh:
                        try:
                            existing = qdrant_client.retrieve(
                                collection_name=QDRANT_COLLECTION,
                                ids=[img_info["path"]],
                            )
                            if existing:
                                scan_status["progress"]["processed"] += 1
                                continue
                        except Exception:
                            pass

                    # 处理图片 (使用 ffmpeg 压缩)
                    image = preprocess_image(img_info["path"])
                    vector = get_image_embedding(image)
                    image.close()

                    # 构建 Rclone URL
                    rclone_url = image_to_rclone_url(img_info["path"])

                    point = qdrant_models.PointStruct(
                        id=img_info["path"],
                        vector={"image": vector},
                        payload={
                            "path": img_info["path"],
                            "rclone_url": rclone_url,
                            "size": img_info["size"],
                            "format": img_info["format"],
                            "indexed_at": datetime.now().isoformat(),
                        },
                    )
                    points.append(point)

                    # 批量提交
                    if len(points) >= BATCH_SIZE:
                        qdrant_client.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=points,
                        )
                        points = []

                    scan_status["progress"]["processed"] = i + 1

                except Exception as e:
                    logger.error(f"处理图片失败 {img_info['path']}: {e}")
                    scan_status["progress"]["failed"] += 1

            # 提交剩余点
            if points:
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                )

            scan_status["last_scan"] = datetime.now().isoformat()
            scan_status["total_indexed"] = scan_status["progress"]["processed"]
            logger.info(f"扫描完成: 成功 {scan_status['progress']['processed']}, 失败 {scan_status['progress']['failed']}")

        except Exception as e:
            logger.error(f"扫描任务失败: {e}")
            traceback.print_exc()
        finally:
            scan_status["is_scanning"] = False

    asyncio.create_task(run_scan())

    # 获取预估数量
    try:
        estimated = len(scan_photos_directory())
    except Exception:
        estimated = 0

    return {
        "task_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "started",
        "message": "目录扫描任务已启动",
        "estimated_count": estimated,
    }


@app.post("/embed/batch")
async def embed_batch(files: List[UploadFile] = File(...)):
    """
    批量 Embedding
    - files: 图片文件列表 (最多 20 个)
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="最多支持 20 个文件")

    load_fashionclip_model()
    init_qdrant()

    errors = []
    success_count = 0

    for file in files:
        try:
            contents = await file.read()
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
            image = resize_image_pil(raw_image)
            if image is not raw_image:
                raw_image.close()
            vector = get_image_embedding(image)
            image.close()

            rclone_url = image_to_rclone_url(f"/tmp/{file.filename}")

            point = qdrant_models.PointStruct(
                id=hashlib.md5(contents).hexdigest(),
                vector={"image": vector},
                payload={
                    "path": f"/tmp/{file.filename}",
                    "rclone_url": rclone_url,
                    "size": len(contents),
                    "format": Path(file.filename).suffix.lower().replace(".", ""),
                    "indexed_at": datetime.now().isoformat(),
                },
            )

            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[point],
            )
            success_count += 1

        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {
        "count": success_count,
        "success": success_count == len(files),
        "errors": errors,
    }


@app.get("/embed/status")
async def get_embed_status():
    """获取扫描状态"""
    try:
        total_indexed = qdrant_client.count(
            collection_name=QDRANT_COLLECTION,
        ).count if qdrant_client else 0
    except Exception:
        total_indexed = scan_status.get("total_indexed", 0)

    return {
        "is_scanning": scan_status["is_scanning"],
        "progress": scan_status["progress"],
        "last_scan": scan_status.get("last_scan"),
        "total_indexed": total_indexed,
    }


@app.get("/embed/{image_path:path}")
async def get_image_embedding_info(image_path: str):
    """获取单张图片的向量信息"""
    init_qdrant()

    try:
        # URL 解码路径
        from urllib.parse import unquote
        image_path = unquote(image_path)

        # 拼接完整路径
        if not image_path.startswith("/"):
            full_path = os.path.join(PHOTOS_DIR, image_path)
        else:
            full_path = image_path

        results = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[full_path],
        )

        if not results:
            raise HTTPException(status_code=404, detail="图片未索引")

        payload = results[0].payload or {}
        return {
            "path": payload.get("path", full_path),
            "vector_dim": 512,
            "indexed_at": payload.get("indexed_at"),
            "rclone_url": payload.get("rclone_url"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取图片信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 启动时初始化 ============
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("FashionCLIP API 服务启动中...")
    load_fashionclip_model()
    init_qdrant()
    logger.info("服务就绪")


# ============ 入口 ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
