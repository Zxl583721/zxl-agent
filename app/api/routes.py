import json
from collections.abc import Generator

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
)
from app.core.config import get_settings
from app.db.session import get_db
from app.models import KnowledgeBase, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.schemas.knowledge import (
    DocumentListResponse,
    KnowledgeDeleteResponse,
    KnowledgeReindexResponse,
    KnowledgeUploadResponse,
)
from app.schemas.task import TaskStatusResponse
from app.services.knowledge_service import KnowledgeService, get_knowledge_service
from app.services.persistence_service import PersistenceService, get_persistence_service
from app.services.rag_service import RAGService, get_rag_service
from app.services.redis_service import RedisService, get_redis_service
from app.services.task_service import TaskService, get_task_service
from app.tasks.celery_app import celery_app


router = APIRouter()


@router.post("/api/auth/register", response_model=TokenResponse, tags=["认证"])
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.scalar(select(User).where(User.username == payload.username))
    if existing is not None:
        raise HTTPException(status_code=409, detail="用户名已存在。")
    if payload.email:
        existing_email = db.scalar(select(User).where(User.email == payload.email))
        if existing_email is not None:
            raise HTTPException(status_code=409, detail="邮箱已存在。")

    user = User(
        username=payload.username,
        email=str(payload.email) if payload.email else None,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name or payload.username,
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.add(KnowledgeBase(user_id=user.id, name="默认知识库", description="User default knowledge base."))
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user), user=UserResponse.from_user(user))


@router.post("/api/auth/login", response_model=TokenResponse, tags=["认证"])
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate_user(db, payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误。")
    return TokenResponse(access_token=create_access_token(user), user=UserResponse.from_user(user))


@router.get("/api/auth/me", response_model=UserResponse, tags=["认证"])
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.from_user(current_user)


@router.post("/api/auth/logout", response_model=LogoutResponse, tags=["认证"])
def logout(_: User = Depends(get_current_user)) -> LogoutResponse:
    return LogoutResponse(message="已退出。请在客户端删除 access token。")


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统状态"],
    summary="健康检查",
)
def health(
    db: Session = Depends(get_db),
    redis: RedisService = Depends(get_redis_service),
) -> dict:
    services = {"api": "ok"}
    try:
        db.execute(text("SELECT 1"))
        services["mysql"] = "ok"
    except SQLAlchemyError:
        services["mysql"] = "error"

    services["redis"] = "ok" if redis.ping() else "error"

    try:
        import chromadb

        chromadb.PersistentClient(path=str(get_settings().vector_store_dir))
        services["chroma"] = "ok"
    except Exception:
        services["chroma"] = "error"

    try:
        services["celery"] = "ok" if celery_app.control.ping(timeout=1) else "error"
    except Exception:
        services["celery"] = "error"

    status_value = "ok" if all(value == "ok" for value in services.values()) else "degraded"
    return {"status": status_value, "services": services}


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["智能问答"],
)
def chat(
    payload: ChatRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(get_rag_service),
    redis: RedisService = Depends(get_redis_service),
    persistence: PersistenceService = Depends(get_persistence_service),
) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入一个问题。")

    knowledge_base_id = resolve_knowledge_base_id(current_user.id, payload.knowledge_base_id, persistence)
    rate_limit = redis.check_chat_rate_limit(current_user.id)
    response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
    response.headers["X-RateLimit-Remaining"] = str(max(rate_limit.limit - rate_limit.current, 0))
    if rate_limit.degraded:
        response.headers["X-RateLimit-Degraded"] = "true"
    if not rate_limit.allowed:
        response.headers["Retry-After"] = str(rate_limit.retry_after)
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试。")

    cached_result = redis.get_cached_chat_answer(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        question=question,
    )
    if cached_result:
        response.headers["X-Cache"] = "HIT"
        return cached_result

    result = rag.chat(
        question=question,
        conversation_id=payload.conversation_id,
        mode=payload.mode,
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
    )
    cache_key = redis.cache_chat_answer(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
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
)
def chat_stream(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(get_rag_service),
    redis: RedisService = Depends(get_redis_service),
    persistence: PersistenceService = Depends(get_persistence_service),
) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入一个问题。")

    knowledge_base_id = resolve_knowledge_base_id(current_user.id, payload.knowledge_base_id, persistence)
    rate_limit = redis.check_chat_rate_limit(current_user.id)
    if not rate_limit.allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试。")

    cached_result = redis.get_cached_chat_answer(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
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
            for event in rag.stream_chat(
                question=question,
                conversation_id=payload.conversation_id,
                mode=payload.mode,
                user_id=current_user.id,
                knowledge_base_id=knowledge_base_id,
            ):
                yield sse_event(str(event.get("type", "message")), event)
        except Exception as exc:
            yield sse_event("error", {"message": f"生成失败：{exc.__class__.__name__}: {exc}"})

    headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "X-RateLimit-Limit": str(rate_limit.limit),
        "X-RateLimit-Remaining": str(max(rate_limit.limit - rate_limit.current, 0)),
        "X-Cache": "HIT" if cached_result else "MISS",
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


@router.post(
    "/api/knowledge/upload",
    response_model=KnowledgeUploadResponse,
    tags=["知识库"],
)
def upload_knowledge(
    files: list[UploadFile] = File(...),
    knowledge_base_id: int | None = Query(default=None, ge=1),
    current_user: User = Depends(get_current_user),
    knowledge: KnowledgeService = Depends(get_knowledge_service),
) -> dict:
    result = knowledge.save_and_index_files(
        files,
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
    )
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
)
def list_documents(
    knowledge_base_id: int | None = Query(default=None, ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    knowledge: KnowledgeService = Depends(get_knowledge_service),
    persistence: PersistenceService = Depends(get_persistence_service),
) -> dict:
    resolved_kb_id = resolve_knowledge_base_id(current_user.id, knowledge_base_id, persistence)
    return knowledge.list_documents(
        user_id=current_user.id,
        knowledge_base_id=resolved_kb_id,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/api/knowledge/documents/{document_id}",
    response_model=KnowledgeDeleteResponse,
    tags=["知识库"],
)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    knowledge: KnowledgeService = Depends(get_knowledge_service),
) -> dict:
    result = knowledge.delete_document(user_id=current_user.id, document_id=document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问。")
    return result


@router.post(
    "/api/knowledge/documents/{document_id}/reindex",
    response_model=KnowledgeReindexResponse,
    tags=["知识库"],
)
def reindex_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    knowledge: KnowledgeService = Depends(get_knowledge_service),
) -> dict:
    result = knowledge.reindex_document(user_id=current_user.id, document_id=document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在、无权访问或文件已丢失。")
    return result


@router.get(
    "/api/conversations/{conversation_id}/messages",
    tags=["智能问答"],
)
def conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    persistence: PersistenceService = Depends(get_persistence_service),
) -> dict:
    result = persistence.conversation_messages(user_id=current_user.id, conversation_id=conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return result


@router.get(
    "/api/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["任务"],
)
def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service),
) -> dict:
    result = task_service.get_task_status(task_id, user_id=current_user.id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return result


def resolve_knowledge_base_id(
    user_id: int,
    requested_knowledge_base_id: int | None,
    persistence: PersistenceService,
) -> int:
    if requested_knowledge_base_id is None:
        return persistence.create_default_knowledge_base_for_user(user_id).id
    if persistence.get_knowledge_base_for_user(user_id, requested_knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在或无权访问。")
    return requested_knowledge_base_id


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
