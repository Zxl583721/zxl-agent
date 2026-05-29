from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: int | None = None
    filename: str
    exists: bool
    indexed: bool
    status: str
    chunk_count: int = 0
    hash: str = ""
    knowledge_base_id: int | None = None
    error_message: str | None = None
    created_at: str | None = None


class DocumentListResponse(BaseModel):
    total: int = 0
    page: int = 1
    page_size: int = 20
    items: list[DocumentInfo] = Field(default_factory=list)


class KnowledgeUploadResponse(BaseModel):
    message: str
    files: list[str] = Field(default_factory=list)
    rejected_files: list[str] = Field(default_factory=list)
    document_ids: list[int] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: int | None = None


class KnowledgeDeleteResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    knowledge_base_id: int


class KnowledgeReindexResponse(BaseModel):
    message: str
    document_id: int
    task_id: str | None = None
