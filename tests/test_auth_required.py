def test_protected_endpoints_require_auth(client):
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 401
    assert client.get("/api/knowledge/documents").status_code == 401
    assert client.post("/api/knowledge/upload", files={"files": ("a.txt", b"hello", "text/plain")}).status_code == 401
