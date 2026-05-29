from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models import ChatMessage, ChatSession, Document, DocumentStatus, KnowledgeBase, TaskRecord, TaskStatus


logger = get_logger(__name__)


class PersistenceService:
    """Database persistence boundary with explicit tenant filters."""

    def create_default_knowledge_base_for_user(self, user_id: int) -> KnowledgeBase:
        with SessionLocal() as db:
            knowledge_base = db.scalar(
                select(KnowledgeBase)
                .where(KnowledgeBase.user_id == user_id)
                .order_by(KnowledgeBase.id.asc())
            )
            if knowledge_base is not None:
                return knowledge_base

            knowledge_base = KnowledgeBase(
                user_id=user_id,
                name="默认知识库",
                description="User default knowledge base.",
            )
            db.add(knowledge_base)
            db.commit()
            db.refresh(knowledge_base)
            return knowledge_base

    def get_knowledge_base_for_user(self, user_id: int, knowledge_base_id: int) -> KnowledgeBase | None:
        with SessionLocal() as db:
            return db.scalar(
                select(KnowledgeBase).where(
                    KnowledgeBase.id == knowledge_base_id,
                    KnowledgeBase.user_id == user_id,
                )
            )

    def create_document_record(
        self,
        *,
        filename: str,
        file_path: Path,
        file_size: int,
        file_hash: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> int | None:
        try:
            with SessionLocal() as db:
                if not self._knowledge_base_exists(db, user_id, knowledge_base_id):
                    return None
                document = Document(
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                    file_path=str(file_path),
                    file_type=file_path.suffix.lower().lstrip("."),
                    file_size=file_size,
                    file_hash=file_hash,
                    status=DocumentStatus.PENDING.value,
                )
                db.add(document)
                db.commit()
                db.refresh(document)
                return document.id
        except SQLAlchemyError:
            logger.exception("Failed to create document record")
            return None

    def list_documents(self, *, user_id: int, knowledge_base_id: int, page: int, page_size: int) -> dict:
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        offset = (page - 1) * page_size

        with SessionLocal() as db:
            base = select(Document).where(
                Document.user_id == user_id,
                Document.knowledge_base_id == knowledge_base_id,
            )
            total = db.scalar(
                select(func.count()).select_from(Document).where(
                    Document.user_id == user_id,
                    Document.knowledge_base_id == knowledge_base_id,
                )
            ) or 0
            documents = db.scalars(
                base.order_by(Document.created_at.desc(), Document.id.desc()).limit(page_size).offset(offset)
            ).all()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "id": document.id,
                    "filename": document.filename,
                    "exists": Path(document.file_path).exists(),
                    "indexed": document.status == DocumentStatus.COMPLETED.value,
                    "status": document.status,
                    "chunk_count": document.chunk_count,
                    "hash": document.file_hash[:12],
                    "knowledge_base_id": document.knowledge_base_id,
                    "error_message": document.error_message,
                    "created_at": document.created_at.isoformat() if document.created_at else None,
                }
                for document in documents
            ],
        }

    def update_document_status(
        self,
        document_id: int | None,
        status: DocumentStatus,
        *,
        user_id: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        if document_id is None:
            return

        try:
            with SessionLocal() as db:
                query = select(Document).where(Document.id == document_id)
                if user_id is not None:
                    query = query.where(Document.user_id == user_id)
                document = db.scalar(query)
                if document is None:
                    return
                document.status = status.value
                if chunk_count is not None:
                    document.chunk_count = chunk_count
                document.error_message = error_message
                db.commit()
        except SQLAlchemyError:
            logger.exception("Failed to update document status document_id=%s", document_id)

    def get_document(self, document_id: int, user_id: int) -> Document | None:
        with SessionLocal() as db:
            return db.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))

    def delete_document(self, *, document_id: int, user_id: int) -> dict | None:
        try:
            with SessionLocal() as db:
                document = db.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))
                if document is None:
                    return None
                result = {
                    "id": document.id,
                    "filename": document.filename,
                    "file_path": document.file_path,
                    "knowledge_base_id": document.knowledge_base_id,
                }
                tasks = db.scalars(
                    select(TaskRecord).where(TaskRecord.document_id == document.id, TaskRecord.user_id == user_id)
                ).all()
                for task in tasks:
                    task.document_id = None
                db.delete(document)
                db.commit()
                return result
        except SQLAlchemyError:
            logger.exception("Failed to delete document document_id=%s", document_id)
            return None

    def reset_document_for_indexing(self, *, document_id: int, user_id: int) -> Document | None:
        try:
            with SessionLocal() as db:
                document = db.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))
                if document is None:
                    return None
                document.status = DocumentStatus.PENDING.value
                document.error_message = None
                document.chunk_count = 0
                db.commit()
                db.refresh(document)
                return document
        except SQLAlchemyError:
            logger.exception("Failed to reset document document_id=%s", document_id)
            return None

    def create_task_record(
        self,
        *,
        task_type: str,
        document_id: int | None,
        user_id: int,
        knowledge_base_id: int,
        task_id: str | None = None,
    ) -> str | None:
        try:
            with SessionLocal() as db:
                if not self._knowledge_base_exists(db, user_id, knowledge_base_id):
                    return None
                task_id = task_id or uuid4().hex
                task = TaskRecord(
                    task_id=task_id,
                    task_type=task_type,
                    status=TaskStatus.PENDING.value,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                )
                db.add(task)
                db.commit()
                return task_id
        except SQLAlchemyError:
            logger.exception("Failed to create task record")
            return None

    def get_task_record(self, task_id: str, user_id: int | None = None) -> dict | None:
        try:
            with SessionLocal() as db:
                query = select(TaskRecord).where(TaskRecord.task_id == task_id)
                if user_id is not None:
                    query = query.where(TaskRecord.user_id == user_id)
                task = db.scalar(query)
                if task is None:
                    return None
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "task_type": task.task_type,
                    "document_id": task.document_id,
                    "user_id": task.user_id,
                    "knowledge_base_id": task.knowledge_base_id,
                    "error_message": task.error_message,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                }
        except SQLAlchemyError:
            logger.exception("Failed to query task record task_id=%s", task_id)
            return None

    def update_task_status(
        self,
        task_id: str | None,
        status: TaskStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        if not task_id:
            return

        try:
            with SessionLocal() as db:
                task = db.scalar(select(TaskRecord).where(TaskRecord.task_id == task_id))
                if task is None:
                    return
                task.status = status.value
                task.error_message = error_message
                db.commit()
        except SQLAlchemyError:
            logger.exception("Failed to update task status task_id=%s", task_id)

    def get_or_create_chat_session(
        self,
        *,
        conversation_id: str | None,
        question: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> ChatSession:
        with SessionLocal() as db:
            if conversation_id:
                session = db.scalar(
                    select(ChatSession).where(
                        ChatSession.conversation_id == conversation_id,
                        ChatSession.user_id == user_id,
                    )
                )
                if session is not None:
                    return session

            session = ChatSession(
                conversation_id=conversation_id or uuid4().hex,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                title=question[:80],
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def recent_messages(self, *, user_id: int, conversation_id: str, limit: int) -> list[dict]:
        with SessionLocal() as db:
            session = db.scalar(
                select(ChatSession).where(
                    ChatSession.conversation_id == conversation_id,
                    ChatSession.user_id == user_id,
                )
            )
            if session is None:
                return []
            messages = db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id, ChatMessage.user_id == user_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
            ).all()

        return [
            {
                "role": message.role,
                "content": message.content,
                "mode": message.mode,
                "sources": json.loads(message.sources_json or "[]"),
            }
            for message in reversed(messages)
        ]

    def conversation_messages(self, *, user_id: int, conversation_id: str) -> dict | None:
        with SessionLocal() as db:
            session = db.scalar(
                select(ChatSession).where(
                    ChatSession.conversation_id == conversation_id,
                    ChatSession.user_id == user_id,
                )
            )
            if session is None:
                return None
            messages = db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id, ChatMessage.user_id == user_id)
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            ).all()

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "mode": message.mode,
                    "sources": json.loads(message.sources_json or "[]"),
                }
                for message in messages
            ],
        }

    def record_chat_exchange(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        mode: str,
        sources: list[dict],
        user_id: int,
        knowledge_base_id: int,
    ) -> None:
        try:
            with SessionLocal() as db:
                session = db.scalar(
                    select(ChatSession).where(
                        ChatSession.conversation_id == conversation_id,
                        ChatSession.user_id == user_id,
                    )
                )
                if session is None:
                    session = ChatSession(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id,
                        title=question[:80],
                    )
                    db.add(session)
                    db.flush()

                sources_json = json.dumps(sources or [], ensure_ascii=False)
                db.add_all(
                    [
                        ChatMessage(
                            session_id=session.id,
                            user_id=user_id,
                            role="user",
                            content=question,
                            mode=mode,
                            sources_json="[]",
                        ),
                        ChatMessage(
                            session_id=session.id,
                            user_id=user_id,
                            role="assistant",
                            content=answer,
                            mode=mode,
                            sources_json=sources_json,
                        ),
                    ]
                )
                db.commit()
        except SQLAlchemyError:
            logger.exception("Failed to record chat exchange")

    @staticmethod
    def _knowledge_base_exists(db: Session, user_id: int, knowledge_base_id: int) -> bool:
        return db.scalar(
            select(KnowledgeBase.id).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        ) is not None


persistence_service = PersistenceService()


def get_persistence_service() -> PersistenceService:
    return persistence_service
