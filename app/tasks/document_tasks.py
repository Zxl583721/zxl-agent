from app.models import DocumentStatus, TaskStatus
from app.services.persistence_service import persistence_service
from app.services.redis_service import redis_service
from app.tasks.celery_app import celery_app
from app.utils.files import kb_collection_name, tenant_data_dir, tenant_vector_dir
from build_index import build_index, load_manifest


@celery_app.task(name="app.tasks.document_tasks.index_document")
def index_document(
    task_id: str,
    document_id: int | None,
    filename: str,
    user_id: int,
    knowledge_base_id: int,
) -> dict:
    """Index a saved document through the existing incremental Chroma pipeline."""
    redis_service.set_task_status(
        task_id,
        TaskStatus.PROCESSING.value,
        {
            "document_id": document_id,
            "filename": filename,
            "task_type": "document_index",
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
        },
    )
    persistence_service.update_task_status(task_id, TaskStatus.PROCESSING)
    persistence_service.update_document_status(document_id, DocumentStatus.PROCESSING, user_id=user_id)

    try:
        if not build_index(
            data_dir=tenant_data_dir(user_id, knowledge_base_id),
            vector_store_dir=tenant_vector_dir(knowledge_base_id),
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            collection_name=kb_collection_name(knowledge_base_id),
        ):
            raise RuntimeError("build_index returned False")

        manifest = load_manifest(tenant_vector_dir(knowledge_base_id) / "index_manifest.json")
        chunk_count = manifest.get("files", {}).get(filename, {}).get("chunk_count", 0)
        persistence_service.update_document_status(
            document_id,
            DocumentStatus.COMPLETED,
            user_id=user_id,
            chunk_count=chunk_count,
        )
        persistence_service.update_task_status(task_id, TaskStatus.COMPLETED)
        payload = {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "task_type": "document_index",
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
        }
        redis_service.set_task_status(task_id, TaskStatus.COMPLETED.value, payload)
        return {"task_id": task_id, "status": TaskStatus.COMPLETED.value, **payload}
    except Exception as exc:
        error_message = f"{exc.__class__.__name__}: {exc}"
        persistence_service.update_document_status(
            document_id,
            DocumentStatus.FAILED,
            user_id=user_id,
            error_message=error_message,
        )
        persistence_service.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error_message=error_message,
        )
        redis_service.set_task_status(
            task_id,
            TaskStatus.FAILED.value,
            {
                "document_id": document_id,
                "filename": filename,
                "task_type": "document_index",
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "error_message": error_message,
            },
        )
        raise
