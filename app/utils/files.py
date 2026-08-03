import os
import re
import zipfile
from pathlib import PurePosixPath
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from xml.etree import ElementTree

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

# `application/octet-stream` is tolerated only for the client-declared MIME:
# browsers and proxies sometimes omit a more precise value.  libmagic must
# still identify the saved bytes as a type that is meaningful for the suffix.
DETECTED_MIME_TYPES = {
    suffix: mime_types - {"application/octet-stream"}
    for suffix, mime_types in ALLOWED_MIME_TYPES.items()
}

# These limits apply to the archive's declared uncompressed entries.  They keep
# a small but highly-compressed Office document from consuming excessive memory
# when it is parsed by python-docx/python-pptx later in the indexing pipeline.
MAX_ARCHIVE_FILE_COUNT = 10_000
MAX_ARCHIVE_COMPRESSION_RATIO = 100
MAX_CONTENT_TYPES_BYTES = 1024 * 1024


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured streaming size limit."""


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
    if not file_path.is_file():
        return False, "invalid_path"
    file_size = file_path.stat().st_size
    if file_size <= 0:
        return False, "empty_file"
    if file_size > settings.max_upload_bytes:
        return False, "file_too_large"

    detected = detect_mime_type(file_path)
    accepted = ALLOWED_MIME_TYPES.get(suffix, set())
    declared_mime = (content_type or "").split(";", 1)[0].strip().lower()
    if declared_mime and declared_mime not in accepted:
        return False, "invalid_mime_type"
    if detected and detected not in DETECTED_MIME_TYPES.get(suffix, set()):
        return False, "invalid_mime_type"
    if not has_valid_file_signature(file_path, suffix):
        return False, "invalid_file_content"
    if not has_valid_file_structure(file_path, suffix):
        return False, "invalid_file_structure"
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
        return header.startswith(b"%PDF-")
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


def has_valid_file_structure(file_path: Path, suffix: str) -> bool:
    """Verify that a file can be parsed as the format claimed by its suffix."""
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path), strict=True)
            return not reader.is_encrypted and len(reader.pages) >= 0
        if suffix == ".docx":
            if not _is_valid_office_archive(file_path, "word/document.xml", "wordprocessingml.document.main+xml"):
                return False
            from docx import Document as DocxDocument

            DocxDocument(str(file_path))
            return True
        if suffix == ".pptx":
            if not _is_valid_office_archive(file_path, "ppt/presentation.xml", "presentationml.presentation.main+xml"):
                return False
            from pptx import Presentation

            Presentation(str(file_path))
            return True
        return suffix == ".txt"
    except Exception:
        # A parser error means the bytes are not a usable instance of the
        # claimed format.  Validation must not turn malformed input into 500s.
        return False


def _is_valid_office_archive(file_path: Path, required_part: str, content_type: str) -> bool:
    settings = get_settings()
    with zipfile.ZipFile(file_path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > MAX_ARCHIVE_FILE_COUNT:
            return False

        total_uncompressed = 0
        for entry in entries:
            normalized_name = entry.filename.replace("\\", "/")
            path = PurePosixPath(normalized_name)
            if path.is_absolute() or ".." in path.parts:
                return False
            total_uncompressed += entry.file_size
            if total_uncompressed > settings.max_archive_uncompressed_bytes:
                return False
            if entry.compress_size and entry.file_size > entry.compress_size * MAX_ARCHIVE_COMPRESSION_RATIO:
                return False

        names = set(archive.namelist())
        if "[Content_Types].xml" not in names or required_part not in names:
            return False
        content_types_info = archive.getinfo("[Content_Types].xml")
        if content_types_info.file_size > MAX_CONTENT_TYPES_BYTES:
            return False
        content_types = ElementTree.fromstring(archive.read(content_types_info))
        return any(
            override.attrib.get("PartName") == f"/{required_part}"
            and override.attrib.get("ContentType") == content_type
            for override in content_types.iter()
            if override.tag.endswith("Override")
        )


def write_upload_to_temp(uploaded_file: UploadFile, temp_dir: Path) -> Path:
    """Stream an upload into a private tenant directory without exceeding its limit."""
    settings = get_settings()
    total_size = 0
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=temp_dir, prefix=".upload-", suffix=".tmp", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := uploaded_file.file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > settings.max_upload_bytes:
                    raise UploadTooLargeError("file_too_large")
                temp_file.write(chunk)
        return temp_path
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        uploaded_file.file.seek(0)


def move_verified_upload_to_storage(temp_path: Path, target_dir: Path, suffix: str) -> Path:
    """Atomically publish a verified upload under an unguessable storage name."""
    for _ in range(5):
        save_path = target_dir / f"{uuid4().hex}{suffix}"
        if not is_safe_child_path(save_path, target_dir):
            raise ValueError("invalid_path")
        try:
            # Both paths are in target_dir. link() therefore publishes without
            # overwriting an existing file; the caller removes the temp link.
            os.link(temp_path, save_path)
            os.chmod(save_path, 0o600)
            temp_path.unlink()
            return save_path
        except FileExistsError:
            continue
    raise RuntimeError("unable_to_allocate_storage_path")
