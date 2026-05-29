from sqlalchemy import select

import app.db.session as session_module
from app.models import User


def test_register_login_and_me(client):
    register_response = client.post(
        "/api/auth/register",
        json={"username": "bob", "password": "password123", "email": "bob@example.com"},
    )
    assert register_response.status_code == 200
    token = register_response.json()["access_token"]

    with session_module.SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == "bob"))
        assert user is not None
        assert user.password_hash != "password123"

    login_response = client.post(
        "/api/auth/login",
        json={"username": "bob", "password": "password123"},
    )
    assert login_response.status_code == 200
    assert login_response.json()["access_token"]

    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "bob"
