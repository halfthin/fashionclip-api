# 第一阶段：构建依赖
FROM python:3.10-slim AS builder

WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 第二阶段：运行镜像
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /code

# 从构建阶段复制 Python 包
COPY --from=builder /install /usr/local

# 安装 Python 运行时 ( slim 镜像没有 Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python

# 复制应用代码
COPY --chown=appuser:appuser app.py /code/app.py

# 创建缓存目录和图片目录
RUN mkdir -p /code/cache && chown -R appuser:appuser /code

# 设置环境变量
ENV PHOTOS=/mnt/dapai-s
ENV RCLONE_BASE_URL=http://192.168.0.10:8080
ENV QDRANT_URL=http://qdrant:6333
ENV QDRANT_COLLECTION=images
ENV DEVICE=cuda
ENV BATCH_SIZE=32
# 图片压缩配置 (可选)
ENV FASHIONCLIP_RESIZE=true
ENV FASHIONCLIP_MAX_DIM=672
ENV FASHIONCLIP_QUALITY=q85

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8008

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]
