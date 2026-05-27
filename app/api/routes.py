from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.schemas.knowledge import DocumentListResponse, KnowledgeUploadResponse
from app.services.knowledge_service import knowledge_service
from app.services.rag_service import rag_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> dict:
    return rag_service.health()


@router.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入一个问题。")
    return rag_service.chat(
        question=question,
        conversation_id=payload.conversation_id,
        mode=payload.mode,
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
    )


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
