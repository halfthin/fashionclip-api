# FashionCLIP 图片相似度搜索 API 文档

基于 FashionCLIP 模型的多模态图片分析和相似搜索服务。

**Swagger UI**: `http://<host>:8008/docs`

## 基础信息

- **服务地址**: `http://<host>:8008`
- **向量维度**: 512 维 (FashionCLIP)
- **模型**: FashionCLIP `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` (512 维向量)
- **向量数据库**: Qdrant

## API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/search` | 搜索相似图片 |
| POST | `/embed/scan` | 触发目录扫描（异步） |
| POST | `/embed/cancel` | 取消正在运行的扫描任务 |
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

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 健康状态: `healthy` |
| `model_loaded` | Boolean | FashionCLIP 模型是否已加载 |
| `device` | String | 运行设备: `cuda` 或 `cpu` |
| `photos_dir` | String | 图片根目录 |
| `qdrant_url` | String | Qdrant 服务地址 |

---

## 2. 搜索相似图片

以图搜图，查找与查询图片最相似的已索引图片。

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
| `threshold` | Float | 否 | 0.0 | 相似度阈值 (0-1)，低于此阈值的 结果不返回 |

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

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_type` | String | 查询方式: `upload` 或 `url` |
| `total` | Integer | 返回结果数量 |
| `results` | Array | 相似图片列表 |
| `results[].path` | String | 图片绝对路径 |
| `results[].rclone_url` | String | Rclone HTTP URL |
| `results[].score` | Float | 相似度分数 (0-1) |
| `results[].size` | Integer | 文件大小 (bytes) |
| `results[].format` | String | 图片格式: `jpg`, `png` |
| `query_time_ms` | Float | 查询耗时 (毫秒) |

---

## 3. 触发目录扫描

扫描指定目录，生成所有图片的向量并存入 Qdrant。**只支持相对路径**，最终路径为 `PHOTOS_DIR/{path}`（即 `/mnt/dapai-s/{path}`）。

扫描为异步操作，可通过 `/embed/status` 查看进度。

### 请求

```
POST /embed/scan
Content-Type: application/x-www-form-urlencoded
```

**参数 (Form Data):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `path` | String | **是** | - | **相对路径**，如 `2026年/3月/0331`，不支持绝对路径 |
| `force_refresh` | Boolean | 否 | false | false=增量扫描, true=全量重建 |

### 示例

```bash
# 扫描指定目录 (相对路径)
curl -X POST http://localhost:8008/embed/scan \
  -d "path=2026%E5%B9%B4/3%E6%9C%88/0331"

# 全量重建
curl -X POST http://localhost:8008/embed/scan \
  -d "path=2026%E5%B9%B4/3%E6%9C%88/0331" \
  -d "force_refresh=true"
```

# 全量重建 (重新处理所有图片)
curl -X POST http://localhost:8008/embed/scan \
  -d "force_refresh=true"
```

### 响应

```json
{
  "task_id": "scan_20260409_030045",
  "status": "started",
  "message": "目录扫描任务已启动",
  "estimated_count": 600
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | String | 任务 ID |
| `status` | String | 状态: `started`, `already_running` |
| `message` | String | 状态消息 |
| `estimated_count` | Integer | 预估待处理图片数量 |

### 扫描进度追踪

```bash
# 查看扫描进度
curl http://localhost:8008/embed/status
```

### Checkpoint 恢复机制

扫描支持断点续传：
- 每处理 100 张图片自动保存 checkpoint 到 `/code/.scan_checkpoint.json`
- 服务重启后可自动恢复扫描进度
- 仅在 `force_refresh=false`（增量模式）时生效

---

## 4. 取消扫描任务

取消正在运行的扫描任务。

### 请求

```
POST /embed/cancel
```

### 示例

```bash
curl -X POST http://localhost:8008/embed/cancel
```

### 响应

```json
{
  "status": "cancel_requested",
  "message": "扫描将在当前批次完成后停止"
}
```

或无任务运行时：

```json
{
  "status": "no_scan_running",
  "message": "没有正在运行的扫描任务"
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

| 字段 | 类型 | 说明 |
|------|------|------|
| `count` | Integer | 成功处理的文件数量 |
| `success` | Boolean | 是否全部成功 |
| `errors` | Array | 错误列表 |

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
    "total": 107962,
    "processed": 5430,
    "failed": 12
  },
  "last_scan": "2026-04-08T07:06:29",
  "total_indexed": 5430
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_scanning` | Boolean | 是否有扫描任务正在运行 |
| `progress.total` | Integer | 总图片数量 |
| `progress.processed` | Integer | 已处理数量 |
| `progress.failed` | Integer | 失败数量 |
| `last_scan` | String | 上次扫描时间 (ISO 格式) |
| `total_indexed` | Integer | Qdrant 中已索引的总数量 |

---

## 7. 获取图片向量信息

根据图片路径查询其向量信息和元数据。

### 请求

```
GET /embed/{image_path}
```

**路径参数:**
- `image_path`: 图片路径 (URL 编码)
  - 如果以 `/` 开头，视为完整路径
  - 否则与 `PHOTOS` 目录拼接

### 示例

```bash
# 查询已索引的图片
curl http://localhost:8008/embed/mnt%2Fdapai-s%2Fcategory%2Fitem001.jpg

# 或使用完整路径
curl http://localhost:8008/embed//mnt/dapai-s/category/item001.jpg
```

### 响应

```json
{
  "path": "/mnt/dapai-s/category/item001.jpg",
  "vector_dim": 512,
  "indexed_at": "2026-03-29T14:25:30",
  "rclone_url": "http://192.168.0.10:8080/category/item001.jpg"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | String | 图片完整路径 |
| `vector_dim` | Integer | 向量维度 (512) |
| `indexed_at` | String | 索引时间 (ISO 格式) |
| `rclone_url` | String | Rclone HTTP URL |

### 错误响应 (404)

```json
{
  "detail": "图片未索引"
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

# 获取扫描状态
def get_scan_status():
    response = requests.get("http://localhost:8008/embed/status")
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

// 获取扫描状态
async function getScanStatus() {
  const response = await fetch('http://localhost:8008/embed/status');
  return response.json();
}

// 取消扫描
async function cancelScan() {
  const response = await fetch('http://localhost:8008/embed/cancel', {
    method: 'POST'
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
| 404 | 图片路径不存在或未索引 |
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

## Point ID 算法

Qdrant 中的 Point ID 使用 **UUID v5** (RFC 4122) 基于文件路径生成，确保跨平台、跨语言去重一致性。

### 算法实现

**Python:**
```python
import uuid

def path_to_point_id(path: str) -> str:
    """将文件路径转换为 Qdrant 合法的 point ID（UUID）"""
    NAMESPACE_DNS = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, path))
```

**TypeScript/JavaScript:**
```typescript
import { v5 as uuidv5 } from 'uuid';

const NAMESPACE_DNS = '6ba7b810-9dad-11d1-80b4-00c04fd430c8';
function pathToPointId(path: string): string {
  return uuidv5(path, NAMESPACE_DNS);
}
```

### 路径约定

- **入库时使用的路径**: 绝对路径（`/mnt/dapai-s/2026年/1月/xxx.jpg`）
- **跨机器使用**: 如需在其他平台计算相同 Point ID，应使用与 `PHOTOS_DIR` 拼接后相同的绝对路径

### 增量扫描去重

扫描时使用增量模式（`force_refresh=false`）会自动跳过已索引的图片：
1. 根据文件路径计算 Point ID
2. 查询 Qdrant 是否已存在该 ID
3. 若存在则跳过，不重复入库

### 示例

```python
# Python
point_id = path_to_point_id("/mnt/dapai-s/2026年/1月/0331/abc.jpg")
# 结果: 例如 "a1b2c3d4-e5f6-..." (UUID 格式)
```

```typescript
// TypeScript
const pointId = pathToPointId("/mnt/dapai-s/2026年/1月/0331/abc.jpg");
// 结果: 与 Python 相同
```

---

## 注意事项

1. **图片格式**: 支持 JPEG、PNG 格式，不支持 WebP
2. **图片大小**: 无明确限制，建议小于 10MB
3. **Qdrant 连接**: 服务启动时必须能连接到 Qdrant，否则会退出
4. **GPU**: 生产环境建议使用 GPU 加速推理（需要 `--gpus all`）
5. **目录扫描**: `/embed/scan` 必须指定 `path` 参数（相对路径），禁止扫描整个 `PHOTOS_DIR`
6. **模型自动下载**: FashionCLIP 模型首次启动时自动从 HuggingFace 下载（使用 hf-mirror.com 镜像）
7. **模型持久化**: 建议挂载 `./data/models:/code/cache` volume 避免重复下载
8. **扫描进度**: 使用 checkpoint 机制，扫描中断可恢复（不同路径使用独立 checkpoint）
9. **信号处理**: 支持 SIGTERM/SIGINT 信号，收到后优雅停止扫描

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PHOTOS` | `/mnt/dapai-s` | 服装图片根目录 |
| `RCLONE_BASE_URL` | `http://192.168.0.10:8080` | Rclone HTTP 服务 |
| `QDRANT_URL` | `http://100.64.0.8:6333` | Qdrant 地址 |
| `QDRANT_COLLECTION` | `images` | Qdrant 集合名 |
| `QDRANT_API_KEY` | - | Qdrant API Key (可选) |
| `DEVICE` | `cuda` | 运行设备 |
| `BATCH_SIZE` | 32 | 批处理大小 |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用图片压缩 |
| `FASHIONCLIP_MAX_DIM` | 672 | 压缩后最大尺寸 |
| `FASHIONCLIP_QUALITY` | 85 | JPEG 压缩质量 |
