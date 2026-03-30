#!/bin/bash
set -e

CACHE_DIR="/code/cache"
MODEL_NAME="laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$CACHE_DIR"

# 检测模型是否已缓存（检查任一快照目录下的模型文件）
MODEL_DIR="$CACHE_DIR/models/models--${MODEL_NAME//\//--}"
if [ -d "$MODEL_DIR/snapshots" ] && find "$MODEL_DIR/snapshots" -name "open_clip_model.safetensors" -type f 2>/dev/null | grep -q .; then
    echo "[fashionclip] Model found in cache at $MODEL_DIR"
else
    echo "[fashionclip] Downloading model from HuggingFace..."
    hf download "$MODEL_NAME" --cache-dir "$CACHE_DIR" --quiet || true
    echo "[fashionclip] Model download complete"
fi

exec uvicorn app:app --host 0.0.0.0 --port 8008
