import app.db.session as session_module
from app.models import Document, DocumentStatus, KnowledgeBase, TaskRecord, TaskStatus, User


def register(client, username):
    response = client.post("/api/auth/register", json={"username": username, "password": "password123"})
    assert response.status_code == 200
    return response.json()


def test_user_cannot_access_other_users_knowledge_base(client):
    user_a = register(client, "tenant_a")
    user_b = register(client, "tenant_b")

    kb_b = None
    with session_module.SessionLocal() as db:
        b = db.query(User).filter_by(username="tenant_b").one()
        kb_b = db.query(KnowledgeBase).filter_by(user_id=b.id).one().id

    response = client.get(
        f"/api/knowledge/documents?knowledge_base_id={kb_b}",
        headers={"Authorization": f"Bearer {user_a['access_token']}"},
    )
    assert response.status_code == 404


def test_user_cannot_access_other_users_task(client):
    user_a = register(client, "task_a")
    user_b = register(client, "task_b")
    with session_module.SessionLocal() as db:
        b = db.query(User).filter_by(username="task_b").one()
        kb = db.query(KnowledgeBase).filter_by(user_id=b.id).one()
        document = Document(
            user_id=b.id,
            knowledge_base_id=kb.id,
            filename="b.txt",
            file_path="/tmp/b.txt",
            file_type="txt",
            file_size=1,
            file_hash="hash",
            status=DocumentStatus.PENDING.value,
        )
        db.add(document)
        db.flush()
        db.add(
            TaskRecord(
                task_id="task-owned-by-b",
                task_type="document_index",
                status=TaskStatus.PENDING.value,
                user_id=b.id,
                knowledge_base_id=kb.id,
                document_id=document.id,
            )
        )
        db.commit()

    response = client.get(
        "/api/tasks/task-owned-by-b",
        headers={"Authorization": f"Bearer {user_a['access_token']}"},
    )
    assert response.status_code == 404
