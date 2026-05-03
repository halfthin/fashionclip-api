# FashionCLIP Embedding API 文档

纯无状态 CLIP 图片向量化服务，不连接任何数据库。

**Swagger UI**: `http://<host>:8008/docs`

## 基础信息

- **服务地址**: `http://<host>:8008`
- **向量维度**: 512 维 (CLIP ViT-B/16)
- **模型**: FashionCLIP `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`
- **状态**: 无状态

## API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/embed` | 单图向量化 |
| POST | `/embed-batch` | 批量向量化（≤20 张） |
| GET | `/health` | 健康检查 |

---

## 1. 健康检查

### 请求

```
GET /health
```

### 响应

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 健康状态: `ok` |
| `model_loaded` | Boolean | 模型是否已加载 |
| `device` | String | 运行设备: `cuda` 或 `cpu` |

---

## 2. 单图向量化

### 请求

```
POST /embed
Content-Type: application/json
```

### 参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `base` | String | 是 | 挂载目录名（如 `dapai-s`），空字符串表示 HTTP URL 模式 |
| `path` | String | 是 | 相对路径或 HTTP URL |

### 路径模式

- **本地模式** (`base` 非空): 拼接为 `/mnt/{base}/{path}`
- **URL 模式** (`base` 为空): `path` 作为 HTTP URL 直接下载

### 示例: 本地文件

```bash
curl -X POST http://localhost:8008/embed \
  -H "Content-Type: application/json" \
  -d '{"base": "dapai-s", "path": "2026年/3月/0331/img.jpg"}'
```

### 示例: HTTP URL

```bash
curl -X POST http://localhost:8008/embed \
  -H "Content-Type: application/json" \
  -d '{"base": "", "path": "http://100.64.0.6:5244/dapai-s/img.jpg"}'
```

### 响应

```json
{
  "embedding": [0.12, -0.34, 0.56, ...]
}
```

### 错误响应

```json
{
  "detail": "文件不存在: /mnt/dapai-s/path/to/img.jpg"
}
```

---

## 3. 批量向量化

### 请求

```
POST /embed-batch
Content-Type: application/json
```

### 参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `base` | String | 是 | 挂载目录名，空字符串表示 HTTP URL 模式 |
| `paths` | String[] | 是 | 路径列表，最多 20 个 |

### 示例: 本地批量

```bash
curl -X POST http://localhost:8008/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"base": "dapai-s", "paths": ["img1.jpg", "img2.jpg"]}'
```

### 示例: URL 批量

```bash
curl -X POST http://localhost:8008/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"base": "", "paths": ["http://example.com/1.jpg", "http://example.com/2.jpg"]}'
```

### 响应

```json
{
  "embeddings": [
    [0.12, -0.34, ...],
    [0.56, 0.78, ...]
  ],
  "errors": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `embeddings` | Number[][] | 向量列表，与输入路径顺序一致 |
| `errors` | Object[] | 处理失败的条目（失败时不整体报错） |

---

## 错误处理

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 400 | 参数错误（base 不合法、路径不存在、图片无法解析） |
| 500 | 服务器内部错误（推理失败） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` | 运行设备 |
| `FASHIONCLIP_RESIZE` | `true` | 是否启用图片缩放 |
| `FASHIONCLIP_MAX_DIM` | `672` | 缩放后最大尺寸 |
| `FASHIONCLIP_QUALITY` | `85` | JPEG 压缩质量 |
