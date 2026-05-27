from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: int | None = None
    filename: str
    exists: bool
    indexed: bool
    status: str
    chunk_count: int = 0
    hash: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)


class KnowledgeUploadResponse(BaseModel):
    message: str
    files: list[str] = Field(default_factory=list)
    rejected_files: list[str] = Field(default_factory=list)
    document_ids: list[int] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
