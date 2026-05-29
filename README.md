# zxl-agent

多用户知识库 RAG/Agent 后端。当前主入口是 FastAPI，提供 JWT 鉴权、用户/知识库隔离、MySQL 持久化、Redis 限流缓存、Celery 异步索引和 Chroma 向量检索。

## 配置

```bash
cp .env.example .env
```

生产环境必须修改：

```bash
MYSQL_PASSWORD=change_me_strong_password
JWT_SECRET_KEY=change_me_to_a_32_plus_char_random_secret
REDIS_PASSWORD=change_me_or_leave_empty_for_local_only
```

如果本地 Redis 没有密码，请把 `REDIS_PASSWORD` 留空。默认模型配置仍使用本机 Ollama：

```bash
ollama pull qwen3:8b
ollama pull bge-m3
```

如需把回答生成模型切到 DeepSeek，在 `.env` 中配置：

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=你的_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 知识库向量化仍保留 Ollama embedding
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
```

修改 `.env` 后重启服务：

```bash
docker compose up -d --build
```

## 启动

推荐用 Docker Compose 分开启动 API、MySQL、Redis 和 Celery worker：

```bash
docker compose up -d
```

服务：

| 服务 | 说明 |
| --- | --- |
| `api` | FastAPI 后端，启动前执行 `python scripts/init_db.py` |
| `mysql` | 保存用户、知识库、文档、会话、消息和任务记录 |
| `redis` | 限流、问答缓存、任务状态缓存、Celery broker/backend |
| `celery_worker` | 文档解析、切分、向量化和 Chroma 写入 |

本地开发也可以直接启动：

```bash
python scripts/init_db.py
uvicorn app.main:app --reload --port 8000
celery -A app.tasks.celery_app worker --loglevel=info --concurrency=1
```

访问：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

`web_app.py` 已废弃，只保留兼容提示；不要再用 Flask 入口启动 Web 服务。

## 注册、登录、上传、问答

注册：

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'
```

登录：

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}' | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

上传文档。未传 `knowledge_base_id` 时会使用当前用户的默认知识库：

```bash
curl -F "files=@data/my_notes.txt" \
  -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/knowledge/upload
```

查看文档分页：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/knowledge/documents?page=1&page_size=20"
```

问答：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"请概括当前知识库内容","mode":"knowledge"}'
```

流式问答：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"请概括当前知识库内容","mode":"knowledge"}'
```

## 安全与隔离

- 所有知识库、文档、聊天和任务接口都需要 JWT Bearer Token。
- 后端只信任 token 中的当前用户，不接受前端传入的 `user_id`。
- 文件保存到 `data/{user_id}/{knowledge_base_id}/`。
- 新索引写入 `vector_store/kb_{knowledge_base_id}` 下的 Chroma collection `kb_{knowledge_base_id}`，metadata 同时保留 `user_id` 和 `knowledge_base_id`。
- 上传文件会检查后缀、MIME type、文件大小、文件名和基础文件头。
- 前端 token 存在 `sessionStorage`，不使用 Cookie 登录态，因此不依赖 Cookie CSRF token；页面继续使用 DOMPurify 清洗 Markdown HTML，重点降低 XSS 泄露 token 风险。

旧的单 collection `personal_knowledge_base` 不自动迁移。历史本地索引需要重新上传文档，或用新的参数化索引流程重建到 `kb_{knowledge_base_id}`。

## 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回：

```json
{
  "status": "ok",
  "services": {
    "api": "ok",
    "mysql": "ok",
    "redis": "ok",
    "chroma": "ok",
    "celery": "ok"
  }
}
```

任一依赖异常时 `status` 为 `degraded`。

## 测试和评估

运行后端测试：

```bash
pytest
```

运行 RAG 质量评估：

```bash
python eval/run_eval.py
```

GitHub Actions 会在 push 和 pull request 时运行 `pytest`。

## 后续优化

- 引入 Alembic 管理正式数据库迁移。
- 增加知识库 CRUD、文档删除和重建索引接口。
- 为 JWT 增加 refresh token 或服务端 blacklist。
- 对 Celery 任务增加更细粒度进度和失败重试策略。
- 对 Chroma 多租户索引增加离线迁移工具。
