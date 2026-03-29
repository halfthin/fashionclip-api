# FashionCLIP 图片相似度搜索 API 文档

基于 FashionCLIP 模型的多模态图片分析和相似搜索服务。

## 基础信息

- **服务地址**: `http://<host>:8008`
- **向量维度**: 512 维 (FashionCLIP)
- **模型**:
  - FashionCLIP: `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` (512 维向量)
  - YOLOv8n-cls: 图像分类 (ImageNet-1k)
  - Segformer B0: 细粒度语义分割
- **向量数据库**: Qdrant

## API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/search` | 搜索相似图片 |
| POST | `/analyze` | 综合图片分析（向量+分类+属性） |
| POST | `/embed/scan` | 触发目录扫描（异步） |
| POST | `/embed/batch` | 批量 embedding |
| GET | `/embed/status` | 获取扫描状态 |
| GET | `/embed/{image_path}` | 获取图片向量信息 |

---

## 1. 健康检查

### 请求

```
GET /health
```

### 响应

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "photos_dir": "/mnt/dapai-s",
  "qdrant_url": "http://100.64.0.8:6333"
}
```

---

## 2. 搜索相似图片

### 请求

```
POST /search
Content-Type: multipart/form-data
```

**参数 (Form Data):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | 与 image_url 二选一 | - | 上传的图片文件 (JPEG/PNG) |
| `image_url` | String | 与 file 二选一 | - | 图片 URL |
| `top_k` | Integer | 否 | 10 | 返回结果数量 |
| `threshold` | Float | 否 | 0.0 | 相似度阈值 (0-1) |

### 示例: 上传图片搜索

```bash
curl -X POST http://localhost:8008/search \
  -F "file=@your_image.jpg" \
  -F "top_k=5"
```

### 示例: URL 图片搜索

```bash
curl -X POST http://localhost:8008/search \
  -F "image_url=https://example.com/image.jpg" \
  -F "top_k=5"
```

### 响应

```json
{
  "query_type": "upload",
  "total": 3,
  "results": [
    {
      "path": "/mnt/dapai-s/category/item001.jpg",
      "rclone_url": "http://192.168.0.10:8080/category/item001.jpg",
      "score": 0.9523,
      "size": 45000,
      "format": "jpg"
    }
  ],
  "query_time_ms": 125.5
}
```

---

## 3. 综合图片分析

综合分析图片，生成 FashionCLIP 向量和多维度分类标签，用于替代阿里百炼的 multimodal embedding。

### 请求

```
POST /analyze
Content-Type: multipart/form-data
```

**参数 (Form Data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 与 image_url 二选一 | 上传的图片文件 |
| `image_url` | String | 与 file 二选一 | 图片 URL |

### 示例

```bash
# 上传图片分析
curl -X POST http://localhost:8008/analyze \
  -F "file=@your_image.jpg"

# URL 图片分析
curl -X POST http://localhost:8008/analyze \
  -F "image_url=https://example.com/image.jpg"
```

### 响应

```json
{
  "embedding": [0.123, -0.456, ...],
  "vector_size": 512,
  "top_class": "jersey",
  "yolo_labels": [
    ["jersey", 0.8923],
    ["cardigan", 0.0451],
    ["sweatshirt", 0.0234],
    ["bikini", 0.0123],
    ["maillot", 0.0089]
  ],
  "detail_labels": [
    [27, 0.1234],
    [15, 0.0891],
    [8, 0.0567],
    [42, 0.0345],
    [3, 0.0212]
  ],
  "combined_text": "jersey(0.89); cardigan(0.05); sweatshirt(0.02); 细粒度: segment_27, segment_15, segment_8, segment_42, segment_3",
  "analyze_time_ms": 345.2
}
```

**字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `embedding` | float[512] | FashionCLIP 512 维归一化向量 |
| `vector_size` | int | 向量维度 (512) |
| `top_class` | string | YOLOv8n-cls 最高置信度分类 |
| `yolo_labels` | array | YOLOv8n-cls top5 分类 [(类别名, 置信度), ...] |
| `detail_labels` | array | Segformer B0 细粒度分割结果 [(类别索引, 比例), ...] |
| `combined_text` | string | 合并后的文本描述 |
| `analyze_time_ms` | float | 分析耗时 (毫秒) |

---

## 4. 触发目录扫描

扫描照片目录，生成所有图片的向量并存入 Qdrant。

### 请求

```
POST /embed/scan
Content-Type: application/x-www-form-urlencoded
```

**参数 (Form Data):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `force_refresh` | Boolean | 否 | false | false=增量扫描, true=全量重建 |

### 示例

```bash
# 增量扫描 (只处理新图片)
curl -X POST http://localhost:8008/embed/scan

# 全量重建
curl -X POST http://localhost:8008/embed/scan \
  -d "force_refresh=true"
```

### 响应

```json
{
  "task_id": "scan_20260329_143052",
  "status": "started",
  "message": "扫描任务已启动"
}
```

---

## 5. 批量 Embedding

将多个图片文件批量转换为向量并存储。

### 请求

```
POST /embed/batch
Content-Type: multipart/form-data
```

**参数 (Form Data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 图片文件列表 (最多 20 个) |

### 示例

```bash
curl -X POST http://localhost:8008/embed/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 响应

```json
{
  "count": 3,
  "success": true,
  "errors": []
}
```

---

## 6. 获取扫描状态

### 请求

```
GET /embed/status
```

### 示例

```bash
curl http://localhost:8008/embed/status
```

### 响应

```json
{
  "is_scanning": false,
  "progress": {
    "total": 1000,
    "processed": 1000,
    "failed": 3
  },
  "last_scan": "2026-03-29T14:30:52",
  "total_indexed": 9997
}
```

---

## 7. 获取图片向量信息

根据图片路径查询其向量信息和元数据。

### 请求

```
GET /embed/{image_path}
```

**路径参数:**
- `image_path`: 图片的完整路径 (URL 编码)

### 示例

```bash
curl http://localhost:8008/embed/mnt%2Fdapai-s%2Fcategory%2Fitem001.jpg
```

### 响应

```json
{
  "path": "/mnt/dapai-s/category/item001.jpg",
  "vector_size": 512,
  "indexed_at": "2026-03-29T14:25:30",
  "exists": true
}
```

---

## 集成示例

### Python 调用

```python
import requests
from PIL import Image
import io

# 上传图片搜索
def search_similar(image_path, top_k=10):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(
            "http://localhost:8008/search",
            files=files,
            data={"top_k": top_k}
        )
    return response.json()

# URL 搜索
def search_by_url(image_url, top_k=10):
    response = requests.post(
        "http://localhost:8008/search",
        data={"image_url": image_url, "top_k": top_k}
    )
    return response.json()

# 获取结果
result = search_similar("/path/to/query.jpg", top_k=5)
for item in result["results"]:
    print(f"{item['path']} - 相似度: {item['score']:.4f}")
```

### JavaScript 调用

```javascript
// 使用 FormData 上传搜索
async function searchSimilar(file, topK = 10) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('top_k', topK);

  const response = await fetch('http://localhost:8008/search', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// 使用 URL 搜索
async function searchByUrl(imageUrl, topK = 10) {
  const formData = new FormData();
  formData.append('image_url', imageUrl);
  formData.append('top_k', topK);

  const response = await fetch('http://localhost:8008/search', {
    method: 'POST',
    body: formData
  });
  return response.json();
}
```

---

## 错误处理

### 常见错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 400 | 请求参数错误 (缺少 file/image_url, 文件格式不支持等) |
| 404 | 图片路径不存在 |
| 500 | 服务器内部错误 (Qdrant 连接失败, 模型加载失败等) |

### 错误响应格式

```json
{
  "detail": "无法解析图片: Cannot identify image format"
}
```

### Python 错误处理示例

```python
import requests

try:
    response = requests.post(
        "http://localhost:8008/search",
        files={"file": open("image.jpg", "rb")},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("请求超时")
except requests.exceptions.HTTPError as e:
    print(f"HTTP 错误: {e.response.status_code}")
    print(e.response.json())
```

---

## 向量兼容性

本服务使用 **CLIP ViT-B/16** 模型生成的 512 维向量，与以下服务兼容:

- `chujiang_alioss_similar` (阿里百炼相似图片服务)

两者共用同一个 Qdrant collection (`images`)，向量可直接对比。

---

## 注意事项

1. **图片格式**: 支持 JPEG、PNG 格式
2. **图片大小**: 无明确限制，建议小于 10MB
3. **Qdrant 连接**: 服务启动时必须能连接到 Qdrant，否则会退出
4. **GPU**: 生产环境建议使用 GPU 加速推理（需要 `--gpus all`）
5. **目录扫描**: `/embed/scan` 为异步操作，返回后扫描仍在后台继续
6. **模型自动下载**: FashionCLIP 模型首次启动时自动从 HuggingFace 下载（使用 hf-mirror.com 镜像）
7. **模型持久化**: 建议挂载 `./data/models:/code/cache` volume 避免重复下载
