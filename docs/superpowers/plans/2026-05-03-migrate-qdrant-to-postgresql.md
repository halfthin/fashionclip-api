# Qdrant 到 PostgreSQL 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除 Qdrant 依赖，将向量存储和搜索迁移到 PostgreSQL + pgvector，清理过时配置。

**Architecture:** 将向量数据库从 Qdrant 替换为 PostgreSQL + pgvector 扩展。使用 `psycopg2-binary` + `pgvector` Python 包，复用现有同步连接模式（与当前 qdrant-client 用法一致）。表结构保持与 Qdrant collection 相同的 payload 字段。

**Tech Stack:** PostgreSQL 16 + pgvector 0.7+, psycopg2-binary, pgvector Python 包

**现状 vs 目标:**

| 项目 | 当前 | 目标 |
|------|------|------|
| 向量数据库 | Qdrant (独立服务) | PostgreSQL + pgvector |
| 连接方式 | `QdrantClient(url, api_key)` | `DATABASE_URL` 连接字符串 |
| 配置项 | `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY` | `DATABASE_URL` |
| Docker 镜像 | Qdrant 需独立部署 | `pgvector/pgvector:pg16` |
| 冗余配置 | `REDIS_URL`（定义但未使用） | 移除 |

---

### Task 1: 更新项目依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 替换依赖**

  移除 `qdrant-client`，添加 `psycopg2-binary` 和 `pgvector`：

  ```txt
  open_clip_torch
  fastapi==0.110.0
  uvicorn[standard]==0.27.0
  torch==2.1.0
  pillow==10.2.0
  numpy==1.26.4
  scikit-learn==1.4.1.post1
  python-dotenv==1.0.1
  psycopg2-binary==2.9.9
  pgvector==0.3.6
  python-multipart==0.0.9
  ```

- [ ] **Step 2: 验证依赖安装**

  Run:
  ```bash
  pip install -r requirements.txt
  ```
  Expected: 无错误，`psycopg2` 和 `pgvector` 可正常 import。

- [ ] **Step 3: Commit**

  ```bash
  git add requirements.txt
  git commit -m "chore: replace qdrant-client with psycopg2-binary and pgvector"
  ```

---

### Task 2: 更新配置文件

**Files:**
- Modify: `.env.example`
- Modify: `.env.docker`

将 QDRANT_* 配置替换为 DATABASE_URL，移除 REDIS_URL。

- [ ] **Step 1: 更新 `.env.example`**

  ```ini
  # ============ 图片目录 ============
  HOST_PORT=8008

  # 服装图片根目录 (容器内路径)
  PHOTOS=/mnt/dapai-s

  # ============ 图片 HTTP URL ============
  # Alist 服务地址，用于将本地路径转换为 HTTP URL
  # URL 格式: {ALIST_BASE_URL}/{root_name}/path/to/file
  ALIST_BASE_URL=http://100.64.0.6:5244

  # ============ PostgreSQL 向量数据库 ============
  DATABASE_URL=postgresql://fashionclip:fashionclip@localhost:5432/fashionclip

  # ============ 图片压缩 (可选) ============
  FASHIONCLIP_RESIZE=true
  FASHIONCLIP_MAX_DIM=672
  FASHIONCLIP_QUALITY=85

  # ============ 运行环境 ============
  DEVICE=cuda
  BATCH_SIZE=32
  ```

- [ ] **Step 2: 更新 `.env.docker`**

  ```ini
  # ============ 图片目录 ============
  HOST_PORT=8008

  PHOTOS=/mnt/dapai-s

  # ============ 图片 HTTP URL ============
  ALIST_BASE_URL=http://100.64.0.6:5244

  # ============ PostgreSQL 向量数据库 ============
  DATABASE_URL=postgresql://fashionclip:fashionclip@postgres:5432/fashionclip

  # ============ 图片压缩 (可选) ============
  FASHIONCLIP_RESIZE=true
  FASHIONCLIP_MAX_DIM=672
  FASHIONCLIP_QUALITY=85

  # ============ 运行环境 ============
  DEVICE=cuda
  BATCH_SIZE=32

  HF_ENDPOINT=https://hf-mirror.com
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add .env.example .env.docker
  git commit -m "chore: replace QDRANT_* config with DATABASE_URL"
  ```

---

### Task 3: 重构 app.py — 替换 Qdrant 为 PostgreSQL

**Files:**
- Modify: `app.py`（全文件修改）

这一任务替换 app.py 中所有 Qdrant 相关代码为 PostgreSQL + pgvector。由于改动涉及文件多个位置，按导入→配置→初始化→增删改查的顺序逐步修改。

**数据库表结构：**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS image_embeddings (
    id UUID PRIMARY KEY,
    embedding vector(512),
    path TEXT NOT NULL,
    image_url TEXT,
    size INTEGER,
    format TEXT,
    indexed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_path
    ON image_embeddings(path);

-- IVFFlat 索引用于余弦相似度近似搜索 (lists=100 适合 10万级数据量)
CREATE INDEX IF NOT EXISTS idx_image_embeddings_embedding
    ON image_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

- [ ] **Step 1: 替换 import 语句**

  移除：
  ```python
  from qdrant_client import QdrantClient
  from qdrant_client.http import models as qdrant_models
  from qdrant_client.http.exceptions import UnexpectedResponse
  ```

  添加：
  ```python
  import psycopg2
  from psycopg2.pool import ThreadedConnectionPool
  from pgvector.psycopg2 import register_vector
  ```

- [ ] **Step 2: 替换配置和全局变量**

  移除（约第 40-78 行）：
  ```python
  PHOTOS_DIR = os.getenv("PHOTOS", "/mnt/dapai-s")
  WXWORK_MEDIA_DIR = "/mnt/wxwork-media"
  IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", os.getenv("RCLONE_BASE_URL", "http://192.168.0.10:8080"))
  ALIST_BASE_URL = os.getenv("ALIST_BASE_URL", "http://100.64.0.6:5244")

  def _parse_scan_roots() -> dict:
      raw = os.getenv("SCAN_ROOTS", "")
      ...

  SCAN_ROOTS = _parse_scan_roots()
  QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
  QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "images")
  QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
  REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
  BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
  DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
  ```

  替换为：
  ```python
  PHOTOS_DIR = os.getenv("PHOTOS", "/mnt/dapai-s")
  IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", os.getenv("RCLONE_BASE_URL", "http://192.168.0.10:8080"))
  ALIST_BASE_URL = os.getenv("ALIST_BASE_URL", "http://100.64.0.6:5244")

  def _parse_scan_roots() -> dict:
      raw = os.getenv("SCAN_ROOTS", "")
      if raw:
          try:
              configs = json.loads(raw)
          except json.JSONDecodeError as e:
              logger.error(f"SCAN_ROOTS JSON 解析失败: {e}")
              raise
      else:
          configs = {
              "dapai-s": "/mnt/dapai-s",
              "wxwork-media": "/mnt/wxwork-media",
              "agentic-outputs": "/mnt/agentic-outputs",
          }
      result = {}
      for name, cfg in configs.items():
          if isinstance(cfg, str):
              result[name] = {"path": cfg, "image_base_url": ""}
          else:
              result[name] = cfg
      return result

  SCAN_ROOTS = _parse_scan_roots()
  DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fashionclip:fashionclip@localhost:5432/fashionclip")
  BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
  DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
  ```

- [ ] **Step 3: 替换全局状态**

  移除：
  ```python
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
  ```

  替换为：
  ```python
  model = None
  preprocess = None
  db_pool = None
  shutdown_requested = False
  scan_status = {
      "is_scanning": False,
      "progress": {"total": 0, "processed": 0, "failed": 0},
      "last_scan": None,
      "total_indexed": 0,
  }
  ```

- [ ] **Step 4: 替换 `init_qdrant()` 为 `init_db()`**

  移除（约第 215-242 行）：
  ```python
  def init_qdrant():
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
  ```

  替换为：
  ```python
  def init_db():
      """初始化 PostgreSQL 连接池，创建表和索引"""
      global db_pool
      if db_pool is None:
          logger.info("正在初始化 PostgreSQL 连接池...")
          db_pool = ThreadedConnectionPool(1, 10, DATABASE_URL)
          conn = db_pool.getconn()
          try:
              with conn.cursor() as cur:
                  cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                  register_vector(conn)
                  cur.execute("""
                      CREATE TABLE IF NOT EXISTS image_embeddings (
                          id UUID PRIMARY KEY,
                          embedding vector(512),
                          path TEXT NOT NULL,
                          image_url TEXT,
                          size INTEGER,
                          format TEXT,
                          indexed_at TIMESTAMP DEFAULT NOW()
                      )
                  """)
                  cur.execute("""
                      CREATE INDEX IF NOT EXISTS idx_image_embeddings_path
                      ON image_embeddings(path)
                  """)
                  cur.execute("""
                      CREATE INDEX IF NOT EXISTS idx_image_embeddings_embedding
                      ON image_embeddings USING ivfflat (embedding vector_cosine_ops)
                      WITH (lists = 100)
                  """)
              conn.commit()
              logger.info("PostgreSQL 表和索引初始化完成")
          except Exception as e:
              logger.error(f"数据库初始化失败: {e}")
              raise
          finally:
              db_pool.putconn(conn)


  def get_conn():
      """从连接池获取一个连接，注册 pgvector 类型"""
      init_db()
      conn = db_pool.getconn()
      register_vector(conn)  # 每个连接都需要注册 vector 类型转换
      return conn


  def put_conn(conn):
      """将连接归还到连接池"""
      if db_pool and conn:
          db_pool.putconn(conn)
  ```

- [ ] **Step 5: 重写 run_scan 中的数据库操作**

  在 `trigger_scan()` 的 `run_scan` 内部（约第 604-802 行）：

  1. **替换 `init_qdrant()` 为 `init_db()`**

  2. **替换增量扫描的重复检查**（约第 656-668 行）：
     移除：
     ```python
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
     ```

     替换为：
     ```python
     if not force_refresh:
         conn = get_conn()
         try:
             with conn.cursor() as cur:
                 cur.execute(
                     "SELECT 1 FROM image_embeddings WHERE id = %s",
                     (path_to_point_id(img_info["path"]),)
                 )
                 if cur.fetchone():
                     scan_status["progress"]["processed"] += 1
                     processed_since_checkpoint += 1
                     continue
         finally:
             put_conn(conn)
     ```

  3. **替换 upsert 写入 Qdrant 的代码**（约第 696-753 行）：
     移除 `qdrant_models.PointStruct` 和 `qdrant_client.upsert`：

     ```python
     # 移除这整段（约 696-753 行）：
     batch_points = []
     if images_data:
         try:
             images = [img for _, img in images_data]
             vectors = get_image_embeddings_batch(images)
             for (img_info_item, image), vector in zip(images_data, vectors):
                 image.close()
                 image_url = image_to_url(img_info_item["path"])
                 point = qdrant_models.PointStruct(...)
                 batch_points.append(point)
         ...

     if batch_points:
         ...
         qdrant_client.upsert(...)
         ...
     ```

     ```python
     if images_data:
         try:
             images = [img for _, img in images_data]
             vectors = get_image_embeddings_batch(images)
             conn = get_conn()
             try:
                 with conn.cursor() as cur:
                     for (img_info_item, image), vector in zip(images_data, vectors):
                         image.close()
                         image_url = image_to_url(img_info_item["path"])
                         cur.execute(
                             """
                             INSERT INTO image_embeddings (id, embedding, path, image_url, size, format, indexed_at)
                             VALUES (%s, %s, %s, %s, %s, %s, NOW())
                             ON CONFLICT (id) DO UPDATE SET
                                 embedding = EXCLUDED.embedding,
                                 path = EXCLUDED.path,
                                 image_url = EXCLUDED.image_url,
                                 size = EXCLUDED.size,
                                 format = EXCLUDED.format,
                                 indexed_at = EXCLUDED.indexed_at
                             """,
                             (
                                 path_to_point_id(img_info_item["path"]),
                                 vector,
                                 img_info_item["path"],
                                 image_url,
                                 img_info_item["size"],
                                 img_info_item["format"],
                             )
                         )
                 conn.commit()
                 scan_status["progress"]["processed"] += len(images_data)
                 processed_since_checkpoint += len(images_data)
             except Exception as e:
                 conn.rollback()
                 logger.error(f"批量写入失败: {e}")
                 scan_status["progress"]["failed"] += len(images_data)
                 processed_since_checkpoint += len(images_data)
             finally:
                 put_conn(conn)
         except Exception as e:
             logger.error(f"批量推理失败: {e}")
             for (img_info_item, image) in images_data:
                 image.close()
                 scan_status["progress"]["failed"] += 1
                 processed_since_checkpoint += 1
     ```

  4. **移除末尾的剩余点提交**（`if points and not shutdown_requested:` 整段，约第 758-772 行）——因为上面已经是逐批写入，不再需要。

- [ ] **Step 6: 重写 `/search` 端点**

  替换（约第 528-563 行）从 `# 搜索 Qdrant` 到格式化结果：

  移除：
  ```python
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
      img_url = payload.get("image_url") or payload.get("rclone_url", "")
      result_item = {
          "path": payload.get("path", ""),
          "score": round(hit.score, 4),
          "size": payload.get("size", 0),
          "format": payload.get("format", ""),
          "image_url": img_url,
      }
      search_results.append(result_item)
  ```

  替换为：
  ```python
  # PostgreSQL 向量搜索
  conn = get_conn()
  try:
      with conn.cursor() as cur:
          register_vector(conn)
          cur.execute(
              """
              SELECT id, path, image_url, size, format, indexed_at,
                     1 - (embedding <=> %s::vector) AS score
              FROM image_embeddings
              WHERE 1 - (embedding <=> %s::vector) >= %s
              ORDER BY embedding <=> %s::vector
              LIMIT %s
              """,
              (query_vector, query_vector, threshold, query_vector, top_k)
          )
          rows = cur.fetchall()
  except Exception as e:
      logger.error(f"PostgreSQL 搜索失败: {e}")
      raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
  finally:
      put_conn(conn)

  search_results = []
  for row in rows:
      record_id, path, image_url, size, fmt, indexed_at, score = row
      result_item = {
          "path": path,
          "score": round(float(score), 4),
          "size": size or 0,
          "format": fmt or "",
          "image_url": image_url or "",
      }
      search_results.append(result_item)
  ```

- [ ] **Step 7: 重写 `/embed/batch` 端点**

  移除 Qdrant 相关代码段（约第 838-880 行），替换为：

  ```python
  @app.post("/embed/batch")
  async def embed_batch(files: List[UploadFile] = File(...)):
      if len(files) > 20:
          raise HTTPException(status_code=400, detail="最多支持 20 个文件")

      load_fashionclip_model()
      init_db()

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

              image_url = image_to_url(f"/tmp/{file.filename}")
              point_id = hashlib.md5(contents).hexdigest()

              conn = get_conn()
              try:
                  with conn.cursor() as cur:
                      register_vector(conn)
                      cur.execute(
                          """
                          INSERT INTO image_embeddings (id, embedding, path, image_url, size, format, indexed_at)
                          VALUES (%s, %s, %s, %s, %s, %s, NOW())
                          ON CONFLICT (id) DO UPDATE SET
                              embedding = EXCLUDED.embedding,
                              image_url = EXCLUDED.image_url,
                              size = EXCLUDED.size,
                              format = EXCLUDED.format,
                              indexed_at = EXCLUDED.indexed_at
                          """,
                          (
                              point_id,
                              vector,
                              f"/tmp/{file.filename}",
                              image_url,
                              len(contents),
                              Path(file.filename).suffix.lower().replace(".", ""),
                          )
                      )
                  conn.commit()
              finally:
                  put_conn(conn)

              success_count += 1

          except Exception as e:
              errors.append({"file": file.filename, "error": str(e)})

      return {
          "count": success_count,
          "success": success_count == len(files),
          "errors": errors,
      }
  ```

- [ ] **Step 8: 重写 `/embed/status` 端点**

  替换（约第 883-898 行）：
  ```python
  @app.get("/embed/status")
  async def get_embed_status():
      try:
          conn = get_conn()
          try:
              with conn.cursor() as cur:
                  cur.execute("SELECT COUNT(*) FROM image_embeddings")
                  total_indexed = cur.fetchone()[0]
          finally:
              put_conn(conn)
      except Exception:
          total_indexed = scan_status.get("total_indexed", 0)

      return {
          "is_scanning": scan_status["is_scanning"],
          "progress": scan_status["progress"],
          "last_scan": scan_status.get("last_scan"),
          "total_indexed": total_indexed,
      }
  ```

- [ ] **Step 9: 重写 `/embed/{image_path}` 端点**

  替换 Qdrant `retrieve` 调用（约第 901-937 行）：

  移除：
  ```python
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
      "image_url": payload.get("image_url") or payload.get("rclone_url"),
  }
  ```

  替换为：
  ```python
  conn = get_conn()
  try:
      with conn.cursor() as cur:
          cur.execute(
              "SELECT path, image_url, indexed_at FROM image_embeddings WHERE id = %s",
              (path_to_point_id(full_path),)
          )
          row = cur.fetchone()
  finally:
      put_conn(conn)

  if not row:
      raise HTTPException(status_code=404, detail="图片未索引")

  return {
      "path": row[0] or full_path,
      "vector_dim": 512,
      "indexed_at": row[2].isoformat() if row[2] else None,
      "image_url": row[1] or "",
  }
  ```

- [ ] **Step 10: 更新 startup 事件**

  替换（约第 941-947 行）：
  ```python
  @app.on_event("startup")
  async def startup_event():
      logger.info("FashionCLIP API 服务启动中...")
      load_fashionclip_model()
      init_qdrant()
      logger.info("服务就绪")
  ```

  替换为：
  ```python
  @app.on_event("startup")
  async def startup_event():
      logger.info("FashionCLIP API 服务启动中...")
      load_fashionclip_model()
      init_db()
      logger.info("服务就绪")
  ```

- [ ] **Step 11: 更新 `/health` 端点**

  替换 health check 返回值（移除 `qdrant_url`）：
  ```python
  @app.get("/health")
  async def health_check():
      model_loaded = model is not None
      return {
          "status": "healthy",
          "model_loaded": model_loaded,
          "device": DEVICE,
          "photos_dir": PHOTOS_DIR,
          "scan_roots": list(SCAN_ROOTS.keys()),
      }
  ```

- [ ] **Step 12: 清理 import（按需）并验证文件完整性**

  确认所有 Qdrant 相关的 import 已移除。文件中不再出现 `qdrant_client`、`qdrant_models`、`UnexpectedResponse`。

- [ ] **Step 13: Commit**

  ```bash
  git add app.py
  git commit -m "refactor: replace Qdrant with PostgreSQL+pgvector for vector storage and search"
  ```

---

### Task 4: 更新 Docker 构建配置

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: 更新 Dockerfile**

  移除 QDRANT_* 和 REDIS_URL 环境变量，添加 DATABASE_URL：

  ```dockerfile
  ENV PHOTOS=/mnt/dapai-s
  ENV ALIST_BASE_URL=http://100.64.0.6:5244
  ENV DATABASE_URL=postgresql://fashionclip:fashionclip@postgres:5432/fashionclip
  ENV DEVICE=cuda
  ENV BATCH_SIZE=32
  ENV FASHIONCLIP_RESIZE=true
  ENV FASHIONCLIP_MAX_DIM=672
  ENV FASHIONCLIP_QUALITY=85
  ENV HF_ENDPOINT=https://hf-mirror.com
  ```

- [ ] **Step 2: 更新 docker-compose.yml**

  添加 PostgreSQL 服务，更新环境变量：

  ```yaml
  services:
    postgres:
      image: pgvector/pgvector:pg16
      container_name: fashionclip-postgres
      restart: always
      environment:
        POSTGRES_DB: fashionclip
        POSTGRES_USER: fashionclip
        POSTGRES_PASSWORD: fashionclip
      volumes:
        - pgdata:/var/lib/postgresql/data
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U fashionclip -d fashionclip"]
        interval: 10s
        timeout: 5s
        retries: 5
      networks:
        - web_net

    fashionclip-api:
      image: halfthin/fashionclip:latest
      container_name: fashionclip-api
      restart: always
      env_file: .env.docker
      ports:
        - "${HOST_PORT:-8008}:8008"
      volumes:
        - /mnt/dapai-s:/mnt/dapai-s:ro
        - /mnt/wxwork-media:/mnt/wxwork-media:ro
        - ./data/models:/code/cache
      depends_on:
        postgres:
          condition: service_healthy
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8008/health"]
        interval: 30s
        timeout: 10s
        retries: 3
        start_period: 60s
      networks:
        - web_net

  volumes:
    pgdata:

  networks:
    web_net:
      external: true
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add Dockerfile docker-compose.yml
  git commit -m "feat: add PostgreSQL service and update Docker config for pgvector"
  ```

---

### Task 5: 更新文档

**Files:**
- Modify: `API.md`
- Modify: `SPEC.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 API.md**

  替换所有 Qdrant 引用：
  - 标题"向量数据库: Qdrant" → "向量数据库: PostgreSQL + pgvector"
  - health 响应中的 `qdrant_url` 字段移除
  - `/embed/scan` 描述中"存入 Qdrant" → "存入 PostgreSQL"
  - `/embed/status` 响应中的 `total_indexed` 说明
  - 环境变量表：移除 `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`, `REDIS_URL`, 添加 `DATABASE_URL`
  - "向量兼容性"段落（第 500-505 行）更新
  - "Point ID 算法"中 Qdrant 描述改为数据库通用描述
  - 错误处理中的 Qdrant → PostgreSQL
  - 注意事项第 3 点更新

- [ ] **Step 2: 更新 SPEC.md**

  替换：
  - 技术栈："向量数据库: Qdrant" → "向量数据库: PostgreSQL + pgvector"
  - 配置表：替换 QDRANT_* 为 DATABASE_URL
  - "向量格式"：存储从 Qdrant collection 改为 PostgreSQL 表
  - 部署方式：移除 QDRANT_URL 环境变量

- [ ] **Step 3: 更新 CLAUDE.md**

  在环境变量表中：
  - 移除 `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`
  - 添加 `DATABASE_URL`
  - 更新"技术栈"中的向量数据库描述
  - 更新"与 chujiang_alioss_similar 的关系"段落

- [ ] **Step 4: Commit**

  ```bash
  git add API.md SPEC.md CLAUDE.md
  git commit -m "docs: update documentation for PostgreSQL migration"
  ```

---

### Task 6: 集成测试验证

- [ ] **Step 1: 启动 PostgreSQL（本地开发）**

  ```bash
  docker run -d --name fashionclip-pg \
    -e POSTGRES_DB=fashionclip \
    -e POSTGRES_USER=fashionclip \
    -e POSTGRES_PASSWORD=fashionclip \
    -p 5432:5432 \
    pgvector/pgvector:pg16
  ```

- [ ] **Step 2: 启动 API 服务**

  ```bash
  python app.py
  ```
  Expected: 日志输出显示"PostgreSQL 表和索引初始化完成"，health 端点返回 status healthy。

- [ ] **Step 3: 测试健康检查**

  ```bash
  curl http://localhost:8008/health
  ```
  期望响应包含 `status: "healthy"` 和 `scan_roots` 列表，无 `qdrant_url`。

- [ ] **Step 4: 测试搜索（无需图片的场景）**

  ```bash
  curl -s http://localhost:8008/search \
    -F "image_url=https://via.placeholder.com/224" \
    -F "top_k=5" | python3 -m json.tool
  ```
  期望：返回 `results` 数组（可能为空），无数据库错误。

- [ ] **Step 5: 清理测试容器**

  ```bash
  docker rm -f fashionclip-pg
  ```

