import io
import zipfile

from app.core.config import get_settings
from app.utils.files import UploadTooLargeError, write_upload_to_temp


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


def test_zip_with_docx_extension_requires_office_structure(client, auth_headers):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("not-a-word-document.txt", "not a docx")

    response = client.post(
        "/api/knowledge/upload",
        headers=auth_headers,
        files={
            "files": (
                "fake.docx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 400
    assert "invalid_file_structure" in response.json()["detail"]["rejected_files"][0]


def test_oversized_upload_is_deleted_before_validation(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 4, raising=False)
    upload = type("Upload", (), {"file": io.BytesIO(b"12345")})()

    try:
        write_upload_to_temp(upload, tmp_path)
    except UploadTooLargeError:
        pass
    else:
        raise AssertionError("oversized upload was accepted")

    assert list(tmp_path.iterdir()) == []


def test_same_display_filename_does_not_overwrite_stored_file(client, auth_headers):
    for content in (b"first", b"second"):
        response = client.post(
            "/api/knowledge/upload",
            headers=auth_headers,
            files={"files": ("notes.txt", content, "text/plain")},
        )
        assert response.status_code == 200

    stored_files = [path for path in get_settings().data_dir.rglob("*") if path.is_file()]
    assert len(stored_files) == 2
    assert len({path.name for path in stored_files}) == 2
    assert all(path.name != "notes.txt" for path in stored_files)
