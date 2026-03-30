FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /code

# 安装 Python 和系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    gosu \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir --no-warn-script-location -r requirements.txt && \
    HOME=/root pip install --no-cache-dir --no-warn-script-location open-clip-torch

# 复制应用代码和启动脚本
COPY --chown=appuser:appuser app.py /code/app.py
COPY --chown=appuser:appuser entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh

# 创建缓存目录
RUN mkdir -p /code/cache /home/appuser/.cache/huggingface && chown -R appuser:appuser /code /home/appuser/.cache

# 设置环境变量
ENV PHOTOS=/mnt/dapai-s
ENV RCLONE_BASE_URL=http://192.168.0.10:8080
ENV QDRANT_URL=http://qdrant:6333
ENV QDRANT_COLLECTION=images
ENV DEVICE=cuda
ENV BATCH_SIZE=32
ENV FASHIONCLIP_RESIZE=true
ENV FASHIONCLIP_MAX_DIM=672
ENV FASHIONCLIP_QUALITY=85
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

ENTRYPOINT ["/code/entrypoint.sh"]
