from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings
from app.models import DocumentStatus, TaskStatus
from app.services.persistence_service import persistence_service
from app.services.redis_service import redis_service
from app.tasks.document_tasks import index_document
from app.utils.files import (
    UploadTooLargeError,
    is_safe_child_path,
    kb_collection_name,
    move_verified_upload_to_storage,
    sanitize_filename,
    tenant_data_dir,
    tenant_vector_dir,
    validate_upload_file,
    write_upload_to_temp,
)
from build_index import delete_documents, get_chroma_vectorstore, hash_file, load_manifest, save_manifest
from src.document_loader import SUPPORTED_EXTENSIONS


class KnowledgeService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def list_documents(self, *, user_id: int, knowledge_base_id: int, page: int, page_size: int) -> dict:
        return persistence_service.list_documents(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
        )

    def delete_document(self, *, user_id: int, document_id: int) -> dict | None:
        document = persistence_service.delete_document(document_id=document_id, user_id=user_id)
        if document is None:
            return None

        knowledge_base_id = int(document["knowledge_base_id"])
        file_path = Path(str(document["file_path"]))
        target_dir = tenant_data_dir(user_id, knowledge_base_id)
        file_entry = None
        if is_safe_child_path(file_path, target_dir) and file_path.exists():
            file_path.unlink()
        if is_safe_child_path(file_path, target_dir):
            manifest_path = tenant_vector_dir(knowledge_base_id) / "index_manifest.json"
            manifest = load_manifest(manifest_path)
            storage_key = file_path.relative_to(target_dir).as_posix()
            file_entry = manifest.get("files", {}).pop(storage_key, None)
            # Documents stored before UUID filenames used the display name as
            # their manifest key. Keep deleting those legacy entries correctly.
            if file_entry is None:
                file_entry = manifest.get("files", {}).pop(document["filename"], None)
        else:
            manifest_path = tenant_vector_dir(knowledge_base_id) / "index_manifest.json"
            manifest = load_manifest(manifest_path)
        if file_entry:
            try:
                vectorstore = get_chroma_vectorstore(
                    vector_store_dir=tenant_vector_dir(knowledge_base_id),
                    collection_name=kb_collection_name(knowledge_base_id),
                )
                delete_documents(vectorstore, file_entry.get("chunk_ids", []))
                save_manifest(manifest, manifest.get("chunk_size") or 400, manifest.get("chunk_overlap") or 80, manifest_path)
            except Exception:
                pass

        return {
            "message": f"已删除文档：{document['filename']}。",
            "document_id": document_id,
            "filename": document["filename"],
            "knowledge_base_id": knowledge_base_id,
        }

    def reindex_document(self, *, user_id: int, document_id: int) -> dict | None:
        document = persistence_service.reset_document_for_indexing(document_id=document_id, user_id=user_id)
        if document is None:
            return None

        file_path = Path(document.file_path)
        if not file_path.exists():
            persistence_service.update_document_status(
                document_id,
                status=DocumentStatus.FAILED,
                user_id=user_id,
                error_message="File is missing from storage.",
            )
            return None

        task_id = uuid4().hex
        task_id = persistence_service.create_task_record(
            task_type="document_index",
            document_id=document.id,
            user_id=user_id,
            knowledge_base_id=document.knowledge_base_id,
            task_id=task_id,
        )
        if task_id:
            payload = {
                "document_id": document.id,
                "filename": document.filename,
                "task_type": "document_index",
                "user_id": user_id,
                "knowledge_base_id": document.knowledge_base_id,
            }
            redis_service.set_task_status(task_id, TaskStatus.PENDING.value, payload)
            index_document.apply_async(
                args=[task_id, document.id, document.filename, user_id, document.knowledge_base_id],
                task_id=task_id,
            )
        return {"message": "已重新投递索引任务。", "document_id": document.id, "task_id": task_id}

    def save_and_index_files(
        self,
        uploaded_files: list[UploadFile],
        *,
        user_id: int,
        knowledge_base_id: int | None,
    ) -> dict:
        if knowledge_base_id is None:
            knowledge_base = persistence_service.create_default_knowledge_base_for_user(user_id)
            knowledge_base_id = knowledge_base.id
        elif persistence_service.get_knowledge_base_for_user(user_id, knowledge_base_id) is None:
            return {
                "ok": False,
                "status_code": 404,
                "message": "知识库不存在或无权访问。",
                "files": [],
                "rejected_files": [],
                "document_ids": [],
                "task_ids": [],
                "knowledge_base_id": knowledge_base_id,
            }

        target_dir = tenant_data_dir(user_id, knowledge_base_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        rejected_files = []
        document_records = []
        task_ids = []

        for uploaded_file in uploaded_files:
            original_name = uploaded_file.filename or ""
            filename = sanitize_filename(original_name)
            suffix = Path(filename).suffix.lower()
            temp_path: Path | None = None

            try:
                temp_path = write_upload_to_temp(uploaded_file, target_dir)
                valid, reason = validate_upload_file(temp_path, filename, uploaded_file.content_type)
                if not filename or suffix not in SUPPORTED_EXTENSIONS or not valid:
                    rejected_files.append(f"{original_name or '未命名文件'}:{reason}")
                    continue

                save_path = move_verified_upload_to_storage(temp_path, target_dir, suffix)
                temp_path = None
                file_hash = hash_file(save_path)
                document_id = persistence_service.create_document_record(
                    filename=filename,
                    file_path=save_path,
                    file_size=save_path.stat().st_size,
                    file_hash=file_hash,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                if document_id is None:
                    save_path.unlink(missing_ok=True)
                    rejected_files.append(f"{original_name or '未命名文件'}:storage_error")
                    continue
                task_id = uuid4().hex
                task_id = persistence_service.create_task_record(
                    task_type="document_index",
                    document_id=document_id,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    task_id=task_id,
                )
                if task_id:
                    payload = {
                        "document_id": document_id,
                        "filename": filename,
                        "task_type": "document_index",
                        "user_id": user_id,
                        "knowledge_base_id": knowledge_base_id,
                    }
                    redis_service.set_task_status(task_id, TaskStatus.PENDING.value, payload)
                    index_document.apply_async(
                        args=[task_id, document_id, filename, user_id, knowledge_base_id],
                        task_id=task_id,
                    )
                saved_files.append(filename)
                document_records.append({"filename": filename, "document_id": document_id, "task_id": task_id})
                if task_id:
                    task_ids.append(task_id)
            except UploadTooLargeError:
                rejected_files.append(f"{original_name or '未命名文件'}:file_too_large")
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

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
                "knowledge_base_id": knowledge_base_id,
            }

        return {
            "ok": True,
            "status_code": 200,
            "message": f"已上传 {len(saved_files)} 个文件，文档索引任务已进入队列。",
            "files": saved_files,
            "rejected_files": rejected_files,
            "document_ids": [record["document_id"] for record in document_records if record["document_id"] is not None],
            "task_ids": task_ids,
            "knowledge_base_id": knowledge_base_id,
        }


knowledge_service = KnowledgeService()


def get_knowledge_service() -> KnowledgeService:
    return knowledge_service
