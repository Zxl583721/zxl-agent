from pydantic import BaseModel


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    task_type: str | None = None
    document_id: int | None = None
    user_id: int | None = None
    knowledge_base_id: int | None = None
    error_message: str | None = None
    redis_cached: bool = False
    created_at: str | None = None
    updated_at: str | None = None

