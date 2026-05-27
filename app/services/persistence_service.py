from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.init_db import seed_defaults
from app.db.session import SessionLocal
from app.models import ChatMessage, ChatSession, Document, DocumentStatus, KnowledgeBase, TaskRecord, TaskStatus, User


class PersistenceService:
    """Database persistence boundary.

    The API should keep serving local RAG responses even while a developer is
    setting up MySQL. Database errors are contained here and reported to callers
    as best-effort failures instead of crashing the app import path.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def ensure_defaults(
        self,
        db: Session,
        *,
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> None:
        seed_defaults(db)
        target_user_id = user_id or self.settings.default_user_id
        target_knowledge_base_id = knowledge_base_id or self.settings.default_knowledge_base_id

        user = db.get(User, target_user_id)
        if user is None:
            db.add(
                User(
                    id=target_user_id,
                    username=f"user_{target_user_id}",
                    display_name=f"User {target_user_id}",
                    is_active=True,
                )
            )

        knowledge_base = db.get(KnowledgeBase, target_knowledge_base_id)
        if knowledge_base is None:
            db.add(
                KnowledgeBase(
                    id=target_knowledge_base_id,
                    user_id=target_user_id,
                    name=f"知识库 {target_knowledge_base_id}",
                    description="Auto-created placeholder knowledge base.",
                )
            )

        db.commit()

    def create_document_record(
        self,
        *,
        filename: str,
        file_path: Path,
        file_size: int,
        file_hash: str,
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> int | None:
        try:
            with SessionLocal() as db:
                self.ensure_defaults(
                    db,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                document = Document(
                    user_id=user_id or self.settings.default_user_id,
                    knowledge_base_id=knowledge_base_id or self.settings.default_knowledge_base_id,
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
        except SQLAlchemyError as exc:
            print(f"数据库写入文档记录失败，已降级为本地索引流程：{exc}")
            return None

    def update_document_status(
        self,
        document_id: int | None,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        if document_id is None:
            return

        try:
            with SessionLocal() as db:
                document = db.get(Document, document_id)
                if document is None:
                    return
                document.status = status.value
                if chunk_count is not None:
                    document.chunk_count = chunk_count
                document.error_message = error_message
                db.commit()
        except SQLAlchemyError as exc:
            print(f"数据库更新文档状态失败，document_id={document_id}：{exc}")

    def create_task_record(
        self,
        *,
        task_type: str,
        document_id: int | None = None,
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> str | None:
        try:
            with SessionLocal() as db:
                self.ensure_defaults(
                    db,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                task_id = uuid4().hex
                task = TaskRecord(
                    task_id=task_id,
                    task_type=task_type,
                    status=TaskStatus.PENDING.value,
                    user_id=user_id or self.settings.default_user_id,
                    knowledge_base_id=knowledge_base_id or self.settings.default_knowledge_base_id,
                    document_id=document_id,
                )
                db.add(task)
                db.commit()
                return task_id
        except SQLAlchemyError as exc:
            print(f"数据库写入任务记录失败，已跳过任务元数据：{exc}")
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
        except SQLAlchemyError as exc:
            print(f"数据库更新任务状态失败，task_id={task_id}：{exc}")

    def record_chat_exchange(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        mode: str,
        sources: list[dict],
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> None:
        try:
            with SessionLocal() as db:
                self.ensure_defaults(
                    db,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                session = self._get_or_create_chat_session(
                    db,
                    conversation_id=conversation_id,
                    question=question,
                    user_id=user_id or self.settings.default_user_id,
                    knowledge_base_id=knowledge_base_id or self.settings.default_knowledge_base_id,
                )
                sources_json = json.dumps(sources or [], ensure_ascii=False)
                db.add_all(
                    [
                        ChatMessage(
                            session_id=session.id,
                            user_id=session.user_id,
                            role="user",
                            content=question,
                            mode=mode,
                            sources_json="[]",
                        ),
                        ChatMessage(
                            session_id=session.id,
                            user_id=session.user_id,
                            role="assistant",
                            content=answer,
                            mode=mode,
                            sources_json=sources_json,
                        ),
                    ]
                )
                db.commit()
        except SQLAlchemyError as exc:
            print(f"数据库写入问答记录失败，已保留 SQLite 会话记录：{exc}")

    @staticmethod
    def _get_or_create_chat_session(
        db: Session,
        *,
        conversation_id: str,
        question: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> ChatSession:
        session = db.scalar(select(ChatSession).where(ChatSession.conversation_id == conversation_id))
        if session is not None:
            return session

        session = ChatSession(
            conversation_id=conversation_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            title=question[:80],
        )
        db.add(session)
        db.flush()
        return session


persistence_service = PersistenceService()
