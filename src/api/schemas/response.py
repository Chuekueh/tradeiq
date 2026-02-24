from typing import Any

from pydantic import BaseModel


class SourceDocument(BaseModel):
    content: str
    source: str
    relevance_score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    cited_sources: list[str]
    session_id: str
    query_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    embedding_model: str
    llm_model: str


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    source: str


class ConversationSession(BaseModel):
    session_id: str
    message_count: int
    last_active: float


class ConversationListResponse(BaseModel):
    sessions: list[ConversationSession]
