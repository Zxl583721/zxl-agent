import app.db.session as session_module
from app.models import Document, DocumentStatus, KnowledgeBase, User


def test_list_documents_pagination(client):
    register_response = client.post("/api/auth/register", json={"username": "pager", "password": "password123"})
    token = register_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with session_module.SessionLocal() as db:
        user = db.query(User).filter_by(username="pager").one()
        kb = db.query(KnowledgeBase).filter_by(user_id=user.id).one()
        for index in range(3):
            db.add(
                Document(
                    user_id=user.id,
                    knowledge_base_id=kb.id,
                    filename=f"doc-{index}.txt",
                    file_path=f"/tmp/doc-{index}.txt",
                    file_type="txt",
                    file_size=10,
                    file_hash=f"hash-{index}",
                    status=DocumentStatus.COMPLETED.value,
                    chunk_count=index,
                )
            )
        db.commit()

    response = client.get("/api/knowledge/documents?page=2&page_size=2", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["page"] == 2
    assert data["page_size"] == 2
    assert len(data["items"]) == 1


def test_page_size_limit(client, auth_headers):
    response = client.get("/api/knowledge/documents?page_size=101", headers=auth_headers)
    assert response.status_code == 422
