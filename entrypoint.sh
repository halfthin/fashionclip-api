#!/bin/bash
set -e

CACHE_DIR="/code/cache"
MODEL_NAME="laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$CACHE_DIR"

# 检测模型是否已缓存
if ls "$HOME/.cache/huggingface/hub/models--${MODEL_NAME//\//--}/snapshots"/*/open_clip_model.safetensors >/dev/null 2>&1; then
    echo "[fashionclip] Model found in cache"
else
    echo "[fashionclip] Downloading model from HuggingFace..."
    hf download "$MODEL_NAME" --cache-dir "$CACHE_DIR" --quiet || true
    echo "[fashionclip] Model download complete"
fi

exec uvicorn app:app --host 0.0.0.0 --port 8008
