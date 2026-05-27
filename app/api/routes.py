from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.schemas.knowledge import DocumentListResponse, KnowledgeUploadResponse
from app.services.knowledge_service import knowledge_service
from app.services.rag_service import rag_service
from app.services.redis_service import redis_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> dict:
    return rag_service.health()


@router.post("/api/chat", response_model=ChatResponse, tags=["chat"])
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


@router.post("/api/knowledge/upload", response_model=KnowledgeUploadResponse, tags=["knowledge"])
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


@router.get("/api/knowledge/documents", response_model=DocumentListResponse, tags=["knowledge"])
def list_documents() -> dict:
    return {"documents": knowledge_service.list_documents()}
