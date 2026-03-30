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
yolo_model = None
segformer_model = None
segformer_processor = None
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
    """加载 FashionCLIP 模型 (OpenCLIP 格式)"""
    global model, preprocess
    if model is None:
        logger.info("正在加载 FashionCLIP 模型...")
        cache_dir = "/code/cache"
        model_path = Path(cache_dir) / "models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K" / "snapshots" / "default"
        safetensors_path = model_path / "open_clip_model.safetensors"

        if safetensors_path.exists():
            # 从本地缓存加载 (通过 hf download 预下载)
            logger.info(f"从本地加载 OpenCLIP 模型: {safetensors_path}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-16",
                pretrained=None,
                checkpoint_path=str(safetensors_path),
            )
        else:
            # 直接从 HuggingFace 加载 (需联网)
            logger.info("从 HuggingFace 加载模型...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-16",
                pretrained="laion2B_s34B_b88K",
            )

        model = model.to(DEVICE)
        model.eval()
        logger.info(f"FashionCLIP 模型加载完成，设备: {DEVICE}")
    return model, preprocess


def load_yolo_model():
    """加载 YOLOv8n-cls 模型"""
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        logger.info("正在加载 YOLOv8n-cls 模型...")
        yolo_model = YOLO("yolov8n-cls.pt")
        if DEVICE == "cuda":
            yolo_model.to(DEVICE)
        logger.info("YOLOv8n-cls 模型加载完成")
    return yolo_model


def classify_with_yolo(image: Image.Image) -> dict:
    """
    使用 YOLOv8n-cls 识别图像主体
    返回: {"top_class": str, "labels": [(label, confidence), ...]}
    """
    m = load_yolo_model()
    # YOLO 需要 numpy array 或图像路径
    img_array = np.array(image.convert("RGB"))
    results = m.predict(img_array, verbose=False)
    probs = results[0].probs
    top5_indices = probs.top5
    top5_conf = probs.top5conf

    labels = []
    for idx, conf in zip(top5_indices, top5_conf):
        labels.append((results[0].names[idx], float(conf)))

    return {
        "top_class": labels[0][0] if labels else None,
        "labels": labels,
    }


def load_segformer_model():
    """加载 Segformer B0 模型"""
    global segformer_model, segformer_processor
    if segformer_model is None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        logger.info("正在加载 Segformer B0 模型...")
        segformer_processor = AutoImageProcessor.from_pretrained(
            "nvidia/mit-b0",
            cache_dir="/code/cache",
        )
        segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            cache_dir="/code/cache",
        )
        segformer_model.to(DEVICE)
        segformer_model.eval()
        logger.info("Segformer B0 模型加载完成")
    return segformer_model, segformer_processor


CLOTHING_DETAIL_KEYWORDS = [
    "collar", "lapel", "neckline", "sleeve", "cuff", "button", "zipper",
    "pocket", "hem", "seam", "fabric", "texture", "pattern", "embroidery",
    "knit", "weave", "lace", "fringe", "ruffle", "pleat", "pleats",
]


def classify_with_segformer(image: Image.Image) -> dict:
    """
    使用 Segformer B0 进行细粒度服装属性识别
    返回: {"segments": [(label, confidence), ...], "detail_text": str}
    """
    m, processor = load_segformer_model()
    img = image.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = m(**inputs)
        logits = outputs.logits  # [1, 150, H, W]

    # 取最大激活的类别
    seg_map = logits.argmax(dim=1)[0].cpu().numpy()
    unique, counts = np.unique(seg_map, return_counts=True)

    # 获取类别标签 (需要了解 MIT-B0 的类别映射，这里使用索引)
    # Segformer MIT-B0 有 150 个 Cityscapes 类别
    # 我们提取高频区域作为 detail
    sorted_segments = sorted(zip(counts, unique), reverse=True)[:10]
    segments = [(int(cls), float(cnt / seg_map.size)) for cnt, cls in sorted_segments]

    # 生成描述文本
    detail_parts = []
    for cls_idx, ratio in sorted_segments[:5]:
        detail_parts.append(f"segment_{cls_idx}")

    detail_text = ", ".join(detail_parts) if detail_parts else "unknown"

    return {
        "segments": segments,
        "detail_text": detail_text,
    }


def analyze_image(image: Image.Image) -> dict:
    """
    综合分析图片: FashionCLIP embedding + YOLOv8 分类 + Segformer 细粒度识别
    """
    # 1. 获取 FashionCLIP embedding
    embedding = get_image_embedding(image)

    # 2. YOLOv8n-cls 分类
    yolo_result = classify_with_yolo(image)
    top_class = yolo_result["top_class"]

    # 3. 如果是服装相关的细分类，使用 Segformer 进一步分析
    detail_result = {}
    if top_class and any(k in str(top_class).lower() for k in ["shirt", "dress", "coat", "jacket", "sweater", "top", "pants", "clothing", "garment"]):
        try:
            detail_result = classify_with_segformer(image)
        except Exception as e:
            logger.warning(f"Segformer 识别失败: {e}")
            detail_result = {"detail_text": "", "segments": []}

    # 4. 合并文本描述
    descriptions = []
    for label, conf in yolo_result["labels"][:3]:
        descriptions.append(f"{label}({conf:.2f})")
    if detail_result.get("detail_text"):
        descriptions.append(f"细粒度: {detail_result['detail_text']}")

    combined_text = "; ".join(descriptions)

    return {
        "embedding": embedding,
        "vector_size": len(embedding),
        "top_class": top_class,
        "yolo_labels": yolo_result["labels"][:5],
        "detail_labels": detail_result.get("segments", [])[:5],
        "combined_text": combined_text,
    }


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
    m, preprocess_fn = load_fashionclip_model()
    with torch.no_grad():
        image = preprocess_fn(image).unsqueeze(0).to(DEVICE)
        image_features = m.encode_image(image)
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


@app.post("/analyze")
async def analyze_image_api(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """
    综合图片分析 (用于替代阿里百炼)

    返回:
    - embedding: FashionCLIP 512 维向量
    - top_class: YOLOv8n-cls 识别的最高置信度类别
    - yolo_labels: YOLOv8n-cls top5 分类结果
    - detail_labels: Segformer B0 细粒度分割结果
    - combined_text: 合并后的文本描述
    """
    start_time = time.time()

    if file:
        try:
            contents = await file.read()
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
            query_image = resize_image_pil(raw_image)
            raw_image.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析图片: {str(e)}")
    elif image_url:
        try:
            import urllib.request
            with urllib.request.urlopen(image_url, timeout=10) as response:
                contents = response.read()
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
            query_image = resize_image_pil(raw_image)
            raw_image.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法下载图片: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="必须提供 file 或 image_url")

    try:
        result = analyze_image(query_image)
    except Exception as e:
        logger.error(f"图片分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")
    finally:
        query_image.close()

    result["analyze_time_ms"] = round((time.time() - start_time) * 1000)
    return result


def extract_prod_code(path: str) -> str:
    """从文件路径中提取款号"""
    import re
    # 款号格式: 款号, code, NO., 数字+字母, 或 N.CODE@ 格式
    patterns = [
        r'款号[：:]\s*([A-Za-z0-9\-_]+)',
        r'code[：:]\s*([A-Za-z0-9\-_]+)',
        r'NO\.?\s*([A-Za-z0-9\-_]+)',
        r'(?:^|[/\-_])([A-Za-z]+\d{5,}[A-Za-z0-9\-_]*)',  # K8610345 格式
        r'(?<=[/\-_.])([A-Za-z]+\d{4,}[A-Za-z0-9\-_]*)(?=@)',  # N.CODE@ 格式
        r'(?:^|[/\-_])([0-9]{4,}[A-Za-z0-9\-_]*)',  # 开头的数字款号
    ]
    for pattern in patterns:
        match = re.search(pattern, path, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


@app.post("/analyze/product")
async def analyze_product(
    folder_path: Optional[str] = Form(None),
    image_urls: Optional[str] = Form(None),
    image_paths: Optional[str] = Form(None),
    images_base64: Optional[str] = Form(None),
):
    """
    款式分析接口 - 支持多种图片输入格式

    输入参数 (至少需要一种):
    - folder_path: 文件夹路径，扫描文件夹下所有图片
    - image_urls: 图片 URL，多个用逗号分隔
    - image_paths: 本地图片路径，多个用逗号分隔
    - images_base64: Base64 编码的图片，格式: [{"name": "xxx.jpg", "data": "base64..."}, ...]

    返回:
    - prod_code: 提取的款号
    - summary: 所有图片识别的汇总
    - detail: 每张图片的详细识别结果
    """
    start_time = time.time()
    import json
    import base64 as b64

    image_sources = []  # [(source_type, source_info, filepath_display)]

    # 1. 处理文件夹路径
    if folder_path:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"文件夹不存在或不是目录: {folder_path}")
        for ext in ALLOWED_EXTENSIONS:
            for img_path in folder.rglob(f"*{ext}"):
                # 跳过缩略图等
                if any(skip in img_path.name.lower() for skip in SKIP_SUBSTRINGS):
                    continue
                image_sources.append(("path", str(img_path), str(img_path)))

    # 2. 处理图片 URL
    if image_urls:
        for url in image_urls.split(","):
            url = url.strip()
            if url:
                image_sources.append(("url", url, url))

    # 3. 处理本地图片路径
    if image_paths:
        for path_str in image_paths.split(","):
            path_str = path_str.strip()
            if path_str:
                p = Path(path_str)
                if not p.exists():
                    logger.warning(f"图片路径不存在，跳过: {path_str}")
                    continue
                image_sources.append(("path", str(p), str(p)))

    # 4. 处理 Base64 图片
    if images_base64:
        try:
            b64_images = json.loads(images_base64)
            for idx, item in enumerate(b64_images):
                name = item.get("name", f"base64_{idx}.jpg")
                data = item.get("data", "")
                image_sources.append(("base64", data, name))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="images_base64 格式错误，应为 JSON 数组")

    if not image_sources:
        raise HTTPException(status_code=400, detail="至少需要提供一种图片来源")

    # 限制数量
    MAX_IMAGES = 50
    if len(image_sources) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"图片数量超过限制，最多 {MAX_IMAGES} 张")

    # 5. 逐个分析图片
    details = []
    all_classes = []
    all_texts = []

    for source_type, source_info, filepath in image_sources:
        try:
            if source_type == "path":
                raw_image = Image.open(source_info).convert("RGB")
                image = resize_image_pil(raw_image)
                raw_image.close()
            elif source_type == "url":
                import urllib.request
                with urllib.request.urlopen(source_info, timeout=10) as response:
                    contents = response.read()
                raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
                image = resize_image_pil(raw_image)
                raw_image.close()
            elif source_type == "base64":
                decoded = b64.b64decode(source_info)
                raw_image = Image.open(io.BytesIO(decoded)).convert("RGB")
                image = resize_image_pil(raw_image)
                raw_image.close()
            else:
                continue

            result = analyze_image(image)
            image.close()

            details.append({
                "filepath": filepath,
                "text": result["combined_text"],
                "top_class": result["top_class"],
            })
            if result["top_class"]:
                all_classes.append(result["top_class"])
            all_texts.append(result["combined_text"])

        except Exception as e:
            logger.error(f"分析图片失败 {filepath}: {e}")
            details.append({
                "filepath": filepath,
                "text": f"[识别失败: {str(e)}]",
                "top_class": None,
            })

    # 6. 提取款号 (从第一个本地路径中提取)
    prod_code = ""
    for src in image_sources:
        if src[0] == "path":
            prod_code = extract_prod_code(src[1])
            if prod_code:
                break

    # 7. 生成汇总 - 将多张图片识别结果合并为产品描述
    summary = generate_product_summary(details, all_classes)

    total_time = round((time.time() - start_time) * 1000)

    return {
        "prod_code": prod_code,
        "summary": summary,
        "detail": details,
        "total_images": len(details),
        "process_time_ms": total_time,
    }


def generate_product_summary(details: list, all_classes: list) -> str:
    """从识别结果生成产品描述"""
    from collections import Counter
    import re

    if not details:
        return "未识别到有效结果"

    # 收集所有标签（只取 YOLO 类别名部分）
    all_labels = []
    for d in details:
        text = d.get("text", "")
        # 分割并提取类别名（不含细粒度部分）
        parts = re.split(r'[;，,]', text)
        for part in parts:
            part = part.strip()
            if "细粒度" in part or part.startswith("segment_"):
                continue
            label = re.sub(r'\([\d.]+\)', '', part).strip()
            if label and not label.startswith("["):
                all_labels.append(label.lower())

    # YOLO 类别到中文描述的映射（处理复合词）
    yolo_category_map = {
        # 上装
        "fur_coat": "毛皮外套", "trench_coat": "风衣", "lab_coat": "实验服",
        "jersey": "运动衫", "sweatshirt": "运动衫", "cardigan": "开衫",
        "hoodie": "卫衣", "sweater": "毛衣", "blouse": "衬衫",
        "shirt": "衬衫", "polo": "POLO衫", "vest": "马甲",
        "t-shirt": "T恤", "suit": "西装", "military_uniform": "军装",
        "gasmask": "防毒面具", "breastplate": "护胸", "cuirass": "护甲",
        # 下装
        "jean": "牛仔裤", "pants": "裤子",
        # 配饰
        "cowboy_hat": "牛仔帽", "punching_bag": "沙袋",
    }

    # 通用关键词列表（用于直接从标签中提取）
    color_keywords = [
        "white", "black", "red", "blue", "green", "yellow", "pink", "purple", "orange", "gray", "grey",
        "brown", "navy", "beige", "cream", "khaki", "burgundy", "maroon", "olive", "mint",
    ]
    detail_keywords = [
        "collar", "lapel", "pocket", "button", "zipper", "hood", "sleeve", "cuff", "hem",
        "round", "v-neck", "crew", "detachable", "lined", "padded",
        "牛角扣", "金属扣", "木扣", "撞色", "压褶", "毛边", "破洞",
        "stripe", "plaid", "pattern", "solid", "print", "embroidery", "lace", "knit",
    ]

    summary_parts = []
    found_categories = set()

    # 第一步：使用映射表转换 YOLO 类别
    for label in all_labels:
        if label in yolo_category_map:
            chinese = yolo_category_map[label]
            if chinese not in found_categories:
                summary_parts.append(chinese)
                found_categories.add(chinese)
        else:
            # 第二步：直接从标签提取颜色和细节关键词
            for kw in color_keywords + detail_keywords:
                if kw.lower() in label:
                    if kw not in found_categories:
                        summary_parts.append(kw)
                        found_categories.add(kw)
                    break

    # 第三步：如果什么都没提取到，使用最常见的类别
    if not summary_parts and all_classes:
        class_counts = Counter(all_classes)
        top_class = class_counts.most_common(1)[0][0] if class_counts else ""
        summary_parts.append(top_class)

    summary = "，".join(summary_parts) if summary_parts else "未识别到有效结果"
    summary += f" ({len(details)}张图片)"

    return summary


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
