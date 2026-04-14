"""
FashionCLIP 图片相似度搜索 API 服务
基于 FashionCLIP 模型，为服饰鞋帽行业提供本地化的图像向量化和搜索服务。
"""

import hashlib
import io
import json
import logging
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
import open_clip

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashionclip")

# ============ 配置 ============
PHOTOS_DIR = os.getenv("PHOTOS", "/mnt/dapai-s")
WXWORK_MEDIA_DIR = "/mnt/wxwork-media"
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
IMAGE_QUALITY = int(os.getenv("FASHIONCLIP_QUALITY", "85"))  # JPEG 质量

# ============ 全局状态 ============
model = None
preprocess = None
qdrant_client = None
shutdown_requested = False
scan_status = {
    "is_scanning": False,
    "progress": {"total": 0, "processed": 0, "failed": 0},
    "last_scan": None,
    "total_indexed": 0,
}

# ============ Checkpoint 配置 ============
CHECKPOINT_DIR = "/code/.scan_checkpoints"
CHECKPOINT_INTERVAL = 100  # 每处理 N 张图片保存一次 checkpoint


def _get_checkpoint_file(scan_path: str) -> str:
    """获取路径特定的 checkpoint 文件"""
    # 将扫描路径转换为安全的文件名
    safe_name = scan_path.replace("/", "_").strip("_") or "root"
    return f"{CHECKPOINT_DIR}/{safe_name}.json"


def save_checkpoint(scan_path: str, data: dict):
    """保存扫描进度 checkpoint"""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(_get_checkpoint_file(scan_path), "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"保存 checkpoint 失败: {e}")


def load_checkpoint(scan_path: str) -> Optional[dict]:
    """加载扫描进度 checkpoint"""
    checkpoint_file = _get_checkpoint_file(scan_path)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载 checkpoint 失败: {e}")
    return None


def clear_checkpoint(scan_path: str):
    """清除 checkpoint"""
    checkpoint_file = _get_checkpoint_file(scan_path)
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except Exception:
            pass


# ============ SIGTERM 信号处理 ============
def handle_shutdown(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info("收到终止信号，将在当前批次完成后停止扫描...")


# 注册信号处理器
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


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
    """加载 FashionCLIP 模型 (OpenCLIP 格式)"""
    global model, preprocess
    if model is None:
        logger.info("正在加载 FashionCLIP 模型...")
        cache_dir = "/code/cache"
        model_base = Path(cache_dir) / "models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K" / "snapshots"

        # 查找实际的快照目录
        safetensors_path = None
        for snap_dir in model_base.iterdir():
            if snap_dir.is_dir():
                candidate = snap_dir / "open_clip_model.safetensors"
                if candidate.exists():
                    safetensors_path = candidate
                    break

        if safetensors_path and safetensors_path.exists():
            # 从本地缓存加载
            logger.info(f"从本地加载 OpenCLIP 模型: {safetensors_path}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-16",
                pretrained=str(safetensors_path),
            )
        else:
            # 直接从 HuggingFace 加载 (需联网)
            logger.info("从 HuggingFace 加载模型...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            )

        model = model.to(DEVICE)
        model.eval()
        logger.info(f"FashionCLIP 模型加载完成，设备: {DEVICE}")
    return model, preprocess


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
            timeout=60,
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
    m, preprocess_fn = load_fashionclip_model()
    with torch.no_grad():
        image = preprocess_fn(image).unsqueeze(0).to(DEVICE)
        image_features = m.encode_image(image)
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()


def path_to_point_id(path: str) -> str:
    """将文件路径转换为 Qdrant 合法的 point ID（UUID）"""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, path))


def get_image_embeddings_batch(images: List[Image.Image]) -> List[List[float]]:
    """批量获取图片的 embedding 向量（GPU 友好）"""
    if not images:
        return []
    m, preprocess_fn = load_fashionclip_model()
    with torch.no_grad():
        tensors = torch.stack([preprocess_fn(img) for img in images]).to(DEVICE)
        image_features = m.encode_image(tensors)
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().tolist()


def preprocess_image_with_path(args: Tuple[str, str]) -> Tuple[str, Optional[Image.Image], str]:
    """预处理单张图片（用于并行处理），返回 (路径, 图片, 错误信息)"""
    path, error_msg = args
    try:
        image = preprocess_image(path)
        return (path, image, "")
    except Exception as e:
        return (path, None, str(e))


def image_to_rclone_url(local_path: str) -> str:
    """将本地路径转换为 Rclone HTTP URL"""
    # /mnt/dapai-s/category/item.jpg -> /category/item.jpg
    if local_path.startswith(PHOTOS_DIR):
        relative_path = local_path[len(PHOTOS_DIR):].lstrip("/")
    else:
        relative_path = local_path.lstrip("/")
    return f"{RCLONE_BASE_URL.rstrip('/')}/{relative_path}"


def scan_photos_directory(relative_path: str = "", root_dir: str = None) -> List[dict]:
    """扫描照片目录，返回所有有效图片信息

    - relative_path: 相对于 root_dir 的路径
    - root_dir: 扫描根目录，默认使用 PHOTOS_DIR
    """
    if root_dir is None:
        root_dir = PHOTOS_DIR
    photos_path = Path(root_dir) / relative_path.strip("/") if relative_path else Path(root_dir)
    if not photos_path.exists():
        raise ValueError(f"照片目录不存在: {photos_path}")

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
        results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            using="image",
            limit=top_k,
            score_threshold=threshold,
        )
    except Exception as e:
        logger.error(f"Qdrant 搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

    # 格式化结果
    search_results = []
    for hit in results.points:
        payload = hit.payload or {}
        result_item = {
            "path": payload.get("path", ""),
            "score": round(hit.score, 4),
            "size": payload.get("size", 0),
            "format": payload.get("format", ""),
        }
        # wxwork-media 路径不返回 rclone_url
        if not payload.get("path", "").startswith(WXWORK_MEDIA_DIR):
            result_item["rclone_url"] = payload.get("rclone_url", "")
        search_results.append(result_item)

    query_time_ms = round((time.time() - start_time) * 1000)

    return {
        "query_type": query_type,
        "results": search_results,
        "total": len(search_results),
        "query_time_ms": query_time_ms,
    }


@app.post("/embed/scan")
async def trigger_scan(path: str = Form(..., description="相对路径，如 '2026年/3月/0331'"),
                      force_refresh: bool = Form(False),
                      base_url: str = Form("dapai-s", description="扫描根目录: 'dapai-s' 或 'wxwork-media'")):
    """
    触发目录扫描 (异步)

    - path: 相对路径，最终拼接为 {base_dir}/{path}
    - force_refresh: false = 增量, true = 全量重新处理
    - base_url: 'dapai-s' -> /mnt/dapai-s, 'wxwork-media' -> /mnt/wxwork-media
    """
    global scan_status, shutdown_requested

    # 验证路径：必须是相对路径，不能是绝对路径或包含 ..
    if os.path.isabs(path) or path.startswith("."):
        raise HTTPException(status_code=400, detail="只支持相对路径，不能以 / 或 . 开头")

    # 根据 base_url 确定扫描根目录
    if base_url == "wxwork-media":
        scan_root = WXWORK_MEDIA_DIR
    else:
        scan_root = PHOTOS_DIR

    # 构建完整扫描路径
    scan_path = Path(scan_root) / path.strip("/")
    if not scan_path.exists():
        raise HTTPException(status_code=400, detail=f"目录不存在: {scan_path}")

    if scan_status["is_scanning"]:
        return {
            "task_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "already_running",
            "message": "扫描任务正在进行中",
        }

    # 启动后台扫描
    import asyncio

    async def run_scan():
        global scan_status, shutdown_requested
        scan_status["is_scanning"] = True
        scan_status["progress"] = {"total": 0, "processed": 0, "failed": 0}

        # 用于 checkpoint 的计数器
        processed_since_checkpoint = 0

        try:
            # 初始化 Qdrant
            init_qdrant()

            # 扫描目录
            image_files = scan_photos_directory(path, scan_root)
            total = len(image_files)
            scan_status["progress"]["total"] = total
            scan_status["progress"]["processed"] = 0
            scan_status["progress"]["failed"] = 0

            # 尝试加载 checkpoint（仅在非 force_refresh 模式）
            checkpoint = None
            start_index = 0
            if not force_refresh:
                checkpoint = load_checkpoint(path)
                if checkpoint and checkpoint.get("total") == total:
                    start_index = checkpoint.get("last_index", 0) + 1
                    scan_status["progress"]["processed"] = checkpoint.get("processed", 0)
                    scan_status["progress"]["failed"] = checkpoint.get("failed", 0)
                    processed_since_checkpoint = scan_status["progress"]["processed"] % CHECKPOINT_INTERVAL
                    logger.info(f"从 checkpoint 恢复: index={start_index}, processed={scan_status['progress']['processed']}")

            logger.info(f"开始扫描 {total} 张图片...")

            # 批量处理 - 并行预处理 + 批量推理
            load_fashionclip_model()
            points = []
            max_workers = min(8, os.cpu_count() or 4)
            i = start_index

            while i < total:
                # 检查是否收到停止信号
                if shutdown_requested:
                    logger.info(f"扫描被中断，已处理到第 {i} 张")
                    break

                # 收集一批图片
                batch_to_check = image_files[i:i + BATCH_SIZE]
                batch_images = []

                for img_info in batch_to_check:
                    # 检查是否已索引 (增量模式)
                    if not force_refresh:
                        try:
                            existing = qdrant_client.retrieve(
                                collection_name=QDRANT_COLLECTION,
                                ids=[path_to_point_id(img_info["path"])],
                            )
                            if existing:
                                scan_status["progress"]["processed"] += 1
                                processed_since_checkpoint += 1
                                continue
                        except Exception:
                            pass
                    batch_images.append(img_info)

                if not batch_images:
                    i += len(batch_to_check)
                    continue

                # 并行预处理
                images_data = []  # (img_info, image)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(preprocess_image, img["path"]): img
                        for img in batch_images
                    }
                    for future in as_completed(futures):
                        img_info_item = futures[future]
                        try:
                            image = future.result()
                            images_data.append((img_info_item, image))
                        except Exception as e:
                            logger.error(f"预处理失败 {img_info_item['path']}: {e}")
                            scan_status["progress"]["failed"] += 1
                            processed_since_checkpoint += 1

                # 批量推理
                batch_points = []
                if images_data:
                    try:
                        images = [img for _, img in images_data]
                        vectors = get_image_embeddings_batch(images)
                        for (img_info_item, image), vector in zip(images_data, vectors):
                            image.close()
                            rclone_url = image_to_rclone_url(img_info_item["path"])
                            point = qdrant_models.PointStruct(
                                id=path_to_point_id(img_info_item["path"]),
                                vector={"image": vector},
                                payload={
                                    "path": img_info_item["path"],
                                    "rclone_url": rclone_url,
                                    "size": img_info_item["size"],
                                    "format": img_info_item["format"],
                                    "indexed_at": datetime.now().isoformat(),
                                },
                            )
                            batch_points.append(point)
                    except Exception as e:
                        logger.error(f"批量推理失败: {e}")
                        for (img_info_item, image) in images_data:
                            image.close()
                            scan_status["progress"]["failed"] += 1
                            processed_since_checkpoint += 1

                # 批量提交 with 重试
                if batch_points:
                    success = False
                    for attempt in range(3):
                        try:
                            qdrant_client.upsert(
                                collection_name=QDRANT_COLLECTION,
                                points=batch_points,
                            )
                            success = True
                            break
                        except Exception as e:
                            if attempt < 2:
                                logger.warning(f"Upsert 失败，重试 ({attempt + 1}/3): {e}")
                                time.sleep(1 * (attempt + 1))
                            else:
                                logger.error(f"Upsert 失败 {len(batch_points)} 个点: {e}")
                                scan_status["progress"]["failed"] += len(batch_points)
                                processed_since_checkpoint += len(batch_points)

                    if success:
                        # 只有 upsert 成功才计入 processed
                        scan_status["progress"]["processed"] += len(batch_points)
                        processed_since_checkpoint += len(batch_points)
                        points.extend(batch_points)

                    # 定期保存 checkpoint
                    if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
                        save_checkpoint(path, {
                            "total": total,
                            "last_index": i,
                            "processed": scan_status["progress"]["processed"],
                            "failed": scan_status["progress"]["failed"],
                        })
                        processed_since_checkpoint = 0

                i += BATCH_SIZE

            # 提交剩余点
            if points and not shutdown_requested:
                for attempt in range(3):
                    try:
                        qdrant_client.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=points,
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.warning(f"Upsert 剩余点失败，重试 ({attempt + 1}/3): {e}")
                            time.sleep(1 * (attempt + 1))
                        else:
                            logger.error(f"Upsert 剩余点失败: {e}")

            # 清理 checkpoint（扫描完成或被中断时）
            if shutdown_requested:
                save_checkpoint(path, {
                    "total": total,
                    "last_index": i - 1,
                    "processed": scan_status["progress"]["processed"],
                    "failed": scan_status["progress"]["failed"],
                })
            else:
                clear_checkpoint(path)

            scan_status["last_scan"] = datetime.now().isoformat()
            scan_status["total_indexed"] = scan_status["progress"]["processed"]
            shutdown_requested = False
            logger.info(f"扫描完成: 成功 {scan_status['progress']['processed']}, 失败 {scan_status['progress']['failed']}")

        except Exception as e:
            logger.error(f"扫描任务失败: {e}")
            traceback.print_exc()
            # 保存 checkpoint 以便恢复
            save_checkpoint(path, {
                "total": scan_status["progress"]["total"],
                "last_index": max(0, i - 1),
                "processed": scan_status["progress"]["processed"],
                "failed": scan_status["progress"]["failed"],
            })
        finally:
            scan_status["is_scanning"] = False

    asyncio.create_task(run_scan())

    # 获取预估数量
    try:
        estimated = len(scan_photos_directory(path, scan_root))
    except Exception:
        estimated = 0

    return {
        "task_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "started",
        "message": "目录扫描任务已启动",
        "estimated_count": estimated,
    }


@app.post("/embed/cancel")
async def cancel_scan():
    """取消正在运行的扫描任务"""
    global shutdown_requested
    if not scan_status["is_scanning"]:
        return {"status": "no_scan_running", "message": "没有正在运行的扫描任务"}
    shutdown_requested = True
    return {"status": "cancel_requested", "message": "扫描将在当前批次完成后停止"}


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
            ids=[path_to_point_id(full_path)],
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
