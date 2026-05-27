from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    conversation_id: str | None = Field(default=None, description="Optional conversation id")
    user_id: int = Field(default=1, ge=1, description="Temporary user id before auth is added")
    knowledge_base_id: int = Field(default=1, ge=1, description="Knowledge base id")
    mode: Literal["knowledge", "general", "chat"] = Field(default="knowledge")


class Source(BaseModel):
    source_id: int | None = None
    source: str | None = None
    chapter: str | None = None
    start_page: str | int | None = None
    end_page: str | int | None = None
    page_range: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    user_id: int = 1
    knowledge_base_id: int = 1
    mode: str = "knowledge"
    cache_hit: bool = False
    cache_key: str | None = None
    sources: list[Source] = Field(default_factory=list)
    citation_status: dict[str, Any] | None = None
    retrieval_question: str | None = None
    rewrite_triggered: bool | None = None
