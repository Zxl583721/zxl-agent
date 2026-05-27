import json
from collections.abc import Generator

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.schemas.knowledge import DocumentListResponse, KnowledgeUploadResponse
from app.schemas.task import TaskStatusResponse
from app.services.knowledge_service import knowledge_service
from app.services.rag_service import rag_service
from app.services.redis_service import redis_service
from app.services.task_service import task_service


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统状态"],
    summary="健康检查",
    description="检查 API 服务、数据库、Redis、知识库索引和 RAG 相关依赖是否可用。",
)
def health() -> dict:
    return rag_service.health()


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["智能问答"],
    summary="发起一次问答",
    description="提交问题并返回完整回答。支持知识库问答和通用聊天模式，会记录会话并写入热点缓存。",
)
def chat(payload: ChatRequest, response: Response) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入一个问题。")

    rate_limit = redis_service.check_chat_rate_limit(payload.user_id)
    response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
    response.headers["X-RateLimit-Remaining"] = str(max(rate_limit.limit - rate_limit.current, 0))
    if rate_limit.degraded:
        response.headers["X-RateLimit-Degraded"] = "true"
    if not rate_limit.allowed:
        response.headers["Retry-After"] = str(rate_limit.retry_after)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "请求过于频繁，请稍后再试。",
                "limit": rate_limit.limit,
                "retry_after": rate_limit.retry_after,
            },
        )

    cached_result = redis_service.get_cached_chat_answer(
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
        question=question,
    )
    if cached_result:
        response.headers["X-Cache"] = "HIT"
        return cached_result

    result = rag_service.chat(
        question=question,
        conversation_id=payload.conversation_id,
        mode=payload.mode,
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
    )
    cache_key = redis_service.cache_chat_answer(
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
        question=question,
        result=result,
    )
    result["cache_hit"] = False
    result["cache_key"] = cache_key
    response.headers["X-Cache"] = "MISS"
    return result


@router.post(
    "/api/chat/stream",
    tags=["智能问答"],
    summary="发起一次流式问答",
    description="提交问题并通过 Server-Sent Events 持续返回生成过程，适合前端边生成边展示。",
)
def chat_stream(payload: ChatRequest, response: Response) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入一个问题。")

    rate_limit = redis_service.check_chat_rate_limit(payload.user_id)
    response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
    response.headers["X-RateLimit-Remaining"] = str(max(rate_limit.limit - rate_limit.current, 0))
    if rate_limit.degraded:
        response.headers["X-RateLimit-Degraded"] = "true"
    if not rate_limit.allowed:
        response.headers["Retry-After"] = str(rate_limit.retry_after)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "请求过于频繁，请稍后再试。",
                "limit": rate_limit.limit,
                "retry_after": rate_limit.retry_after,
            },
        )

    cached_result = redis_service.get_cached_chat_answer(
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
        question=question,
    )

    def generate() -> Generator[str, None, None]:
        if cached_result:
            yield sse_event(
                "metadata",
                {
                    "conversation_id": cached_result.get("conversation_id"),
                    "mode": cached_result.get("mode", payload.mode),
                    "sources": cached_result.get("sources", []),
                    "cache_hit": True,
                    "cache_key": cached_result.get("cache_key"),
                },
            )
            yield sse_event("token", {"content": cached_result.get("answer", "")})
            yield sse_event("done", {"conversation_id": cached_result.get("conversation_id")})
            return

        try:
            for event in rag_service.stream_chat(
                question=question,
                conversation_id=payload.conversation_id,
                mode=payload.mode,
                user_id=payload.user_id,
                knowledge_base_id=payload.knowledge_base_id,
            ):
                yield sse_event(str(event.get("type", "message")), event)
        except Exception as exc:
            yield sse_event("error", {"message": f"生成失败：{exc.__class__.__name__}: {exc}"})

    headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "X-RateLimit-Limit": str(rate_limit.limit),
        "X-RateLimit-Remaining": str(max(rate_limit.limit - rate_limit.current, 0)),
    }
    if rate_limit.degraded:
        headers["X-RateLimit-Degraded"] = "true"
    if cached_result:
        headers["X-Cache"] = "HIT"
    else:
        headers["X-Cache"] = "MISS"
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


@router.post(
    "/api/knowledge/upload",
    response_model=KnowledgeUploadResponse,
    tags=["知识库"],
    summary="上传知识库文件",
    description="上传一个或多个文件，系统会保存文件并提交后台任务进行解析、切分、向量化和索引写入。",
)
def upload_knowledge(files: list[UploadFile] = File(...)) -> dict:
    result = knowledge_service.save_and_index_files(files)
    if not result["ok"]:
        raise HTTPException(
            status_code=result["status_code"],
            detail={
                "message": result["message"],
                "files": result["files"],
                "rejected_files": result["rejected_files"],
            },
        )
    return result


@router.get(
    "/api/knowledge/documents",
    response_model=DocumentListResponse,
    tags=["知识库"],
    summary="查看知识库文档",
    description="返回当前知识库中已经保存和索引的文档列表。",
)
def list_documents() -> dict:
    return {"documents": knowledge_service.list_documents()}


@router.get(
    "/api/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["任务"],
    summary="查询后台任务状态",
    description="根据任务 ID 查询文件解析、向量化、索引写入等后台任务的执行状态。",
)
def get_task(task_id: str) -> dict:
    result = task_service.get_task_status(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return result


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
