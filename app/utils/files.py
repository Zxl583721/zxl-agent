import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from app.core.config import get_settings
from src.document_loader import SUPPORTED_EXTENSIONS


ALLOWED_MIME_TYPES = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/zip",
        "application/octet-stream",
    },
}

def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", name)
    return name.strip(" .")


def is_safe_child_path(file_path: Path, parent_dir: Path) -> bool:
    try:
        file_path.resolve().relative_to(parent_dir.resolve())
    except ValueError:
        return False
    return True


def tenant_data_dir(user_id: int, knowledge_base_id: int) -> Path:
    settings = get_settings()
    return settings.data_dir / str(user_id) / str(knowledge_base_id)


def tenant_vector_dir(knowledge_base_id: int) -> Path:
    settings = get_settings()
    return settings.vector_store_dir / f"kb_{knowledge_base_id}"


def kb_collection_name(knowledge_base_id: int) -> str:
    return f"kb_{knowledge_base_id}"


def validate_upload_file(file_path: Path, original_filename: str, content_type: str | None) -> tuple[bool, str]:
    filename = sanitize_filename(original_filename)
    suffix = Path(filename).suffix.lower()
    settings = get_settings()

    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        return False, "unsupported_extension"
    if not is_safe_child_path(file_path, file_path.parent):
        return False, "invalid_path"
    file_size = file_path.stat().st_size
    if file_size <= 0:
        return False, "empty_file"
    if file_size > settings.max_upload_bytes:
        return False, "file_too_large"

    detected = detect_mime_type(file_path)
    accepted = ALLOWED_MIME_TYPES.get(suffix, set())
    if content_type and content_type not in accepted:
        return False, "invalid_mime_type"
    if detected and detected not in accepted:
        return False, "invalid_mime_type"
    if not has_valid_file_signature(file_path, suffix):
        return False, "invalid_file_content"
    return True, "ok"


def detect_mime_type(file_path: Path) -> str | None:
    try:
        import magic
    except ImportError:
        return None
    try:
        return str(magic.from_file(str(file_path), mime=True))
    except Exception:
        return None


def has_valid_file_signature(file_path: Path, suffix: str) -> bool:
    with file_path.open("rb") as file:
        header = file.read(8)

    if suffix == ".pdf":
        return header.startswith(b"%PDF")
    if suffix in {".docx", ".pptx"}:
        return header.startswith(b"PK\x03\x04")
    if suffix == ".txt":
        content = file_path.read_bytes()
        if not content.strip():
            return False
        for encoding in ("utf-8", "gbk"):
            try:
                content.decode(encoding)
                return True
            except UnicodeDecodeError:
                continue
        return False
    return False


def write_upload_to_temp(uploaded_file: UploadFile) -> Path:
    settings = get_settings()
    total_size = 0
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        while True:
            chunk = uploaded_file.file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.max_upload_bytes:
                temp_file.write(chunk)
                break
            temp_file.write(chunk)
    uploaded_file.file.seek(0)
    return temp_path
