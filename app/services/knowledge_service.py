import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings
from app.models import TaskStatus
from app.services.persistence_service import persistence_service
from app.services.redis_service import redis_service
from app.tasks.document_tasks import index_document
from app.utils.files import sanitize_filename
from build_index import DATA_DIR, discover_knowledge_files, hash_file, load_manifest
from src.document_loader import SUPPORTED_EXTENSIONS


class KnowledgeService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def list_documents(self) -> list[dict]:
        manifest = load_manifest()
        indexed_files = manifest.get("files", {})
        files_by_name = {file_path.name: file_path for file_path in discover_knowledge_files(DATA_DIR)}
        all_names = sorted(set(files_by_name) | set(indexed_files))
        documents = []

        for filename in all_names:
            file_path = files_by_name.get(filename)
            manifest_entry = indexed_files.get(filename, {})
            exists = file_path is not None and file_path.exists()
            current_hash = hash_file(file_path) if exists else ""
            indexed_hash = manifest_entry.get("hash", "")
            indexed = bool(manifest_entry)

            documents.append(
                {
                    "filename": filename,
                    "exists": exists,
                    "indexed": indexed,
                    "status": self._document_status(exists, indexed, current_hash, indexed_hash),
                    "chunk_count": manifest_entry.get("chunk_count", 0),
                    "hash": current_hash[:12] if current_hash else indexed_hash[:12],
                }
            )

        return documents

    def save_and_index_files(self, uploaded_files: list[UploadFile]) -> dict:
        saved_files = []
        rejected_files = []
        document_records = []
        task_ids = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            original_name = uploaded_file.filename or ""
            filename = sanitize_filename(original_name)
            suffix = Path(filename).suffix.lower()

            if not filename or suffix not in SUPPORTED_EXTENSIONS:
                rejected_files.append(original_name or "未命名文件")
                continue

            save_path = DATA_DIR / filename
            with save_path.open("wb") as output:
                shutil.copyfileobj(uploaded_file.file, output)
            file_hash = hash_file(save_path)
            document_id = persistence_service.create_document_record(
                filename=filename,
                file_path=save_path,
                file_size=save_path.stat().st_size,
                file_hash=file_hash,
            )
            task_id = uuid4().hex
            task_id = persistence_service.create_task_record(
                task_type="document_index",
                document_id=document_id,
                task_id=task_id,
            )
            if task_id:
                redis_service.set_task_status(
                    task_id,
                    TaskStatus.PENDING.value,
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "task_type": "document_index",
                    },
                )
                index_document.apply_async(
                    args=[task_id, document_id, filename],
                    task_id=task_id,
                )
            saved_files.append(filename)
            document_records.append({"filename": filename, "document_id": document_id, "task_id": task_id})
            if task_id:
                task_ids.append(task_id)

        if not saved_files:
            supported_formats = "、".join(sorted(SUPPORTED_EXTENSIONS))
            return {
                "ok": False,
                "status_code": 400,
                "message": f"没有可入库的文件。当前支持：{supported_formats}。",
                "files": saved_files,
                "rejected_files": rejected_files,
                "document_ids": [],
                "task_ids": [],
            }

        return {
            "ok": True,
            "status_code": 200,
            "message": f"已上传 {len(saved_files)} 个文件，文档索引任务已进入队列。",
            "files": saved_files,
            "rejected_files": rejected_files,
            "document_ids": [record["document_id"] for record in document_records if record["document_id"] is not None],
            "task_ids": task_ids,
        }

    @staticmethod
    def _document_status(exists: bool, indexed: bool, current_hash: str, indexed_hash: str) -> str:
        if not exists and indexed:
            return "deleted"
        if exists and not indexed:
            return "new"
        if exists and indexed and current_hash != indexed_hash:
            return "changed"
        if exists and indexed:
            return "indexed"
        return "unknown"


knowledge_service = KnowledgeService()
