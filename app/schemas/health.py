from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    ready: bool
    has_index: bool
    document_count: int
    message: str

