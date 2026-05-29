def test_illegal_extension_rejected(client, auth_headers):
    response = client.post(
        "/api/knowledge/upload",
        headers=auth_headers,
        files={"files": ("evil.exe", b"hello", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "rejected_files" in response.json()["detail"]


def test_illegal_mime_rejected(client, auth_headers):
    response = client.post(
        "/api/knowledge/upload",
        headers=auth_headers,
        files={"files": ("fake.pdf", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_path_traversal_filename_is_sanitized(client, auth_headers):
    response = client.post(
        "/api/knowledge/upload",
        headers=auth_headers,
        files={"files": ("../notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["files"] == ["notes.txt"]
