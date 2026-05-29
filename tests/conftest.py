import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_with_more_than_32_chars")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.api.routes import celery_app
from app.db import migrations
from app.db.base import Base
from app.db.session import get_db
from app.main import app


class FakeRateLimit:
    allowed = True
    key = "fake"
    current = 1
    limit = 100
    retry_after = 0
    degraded = False


class FakeRedisService:
    def ping(self):
        return True

    def check_chat_rate_limit(self, user_id):
        return FakeRateLimit()

    def get_cached_chat_answer(self, **kwargs):
        return None

    def cache_chat_answer(self, **kwargs):
        return None

    def set_task_status(self, *args, **kwargs):
        return None

    def get_task_status(self, task_id):
        return None


class FakeRAGService:
    def chat(self, *, question, conversation_id, mode, user_id, knowledge_base_id):
        return {
            "answer": f"answer: {question}",
            "conversation_id": conversation_id or "test-conversation",
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
            "mode": mode,
            "sources": [],
        }

    def stream_chat(self, *, question, conversation_id, mode, user_id, knowledge_base_id):
        yield {"type": "metadata", "conversation_id": conversation_id or "test-conversation", "sources": []}
        yield {"type": "token", "content": f"answer: {question}"}
        yield {"type": "done", "conversation_id": conversation_id or "test-conversation"}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    test_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    import app.api.routes as routes
    import app.main as main_module
    import app.db.session as session_module
    import app.services.persistence_service as persistence_module
    import app.services.knowledge_service as knowledge_module
    from app.core.config import get_settings
    from app.services.rag_service import get_rag_service
    from app.services.redis_service import get_redis_service

    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data", raising=False)
    monkeypatch.setattr(settings, "vector_store_dir", tmp_path / "vector_store", raising=False)
    monkeypatch.setattr(settings, "max_upload_bytes", 1024 * 1024, raising=False)
    monkeypatch.setattr(main_module, "engine", test_engine)
    monkeypatch.setattr(migrations, "engine", test_engine)
    monkeypatch.setattr(session_module, "engine", test_engine)
    monkeypatch.setattr(session_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(persistence_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(routes.celery_app.control, "ping", lambda timeout=1: [{"worker": "pong"}])
    monkeypatch.setattr(knowledge_module.index_document, "apply_async", lambda *args, **kwargs: None)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis_service] = lambda: FakeRedisService()
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client):
    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "password123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
