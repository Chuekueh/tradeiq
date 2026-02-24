import uuid

from fastapi import APIRouter, HTTPException

from src.api.dependencies import app_state
from src.api.schemas.request import QueryRequest
from src.api.schemas.response import QueryResponse, SourceDocument

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not app_state.rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    session_id = request.session_id or str(uuid.uuid4())
    where = request.filters.to_chroma_where() if request.filters else None

    response = app_state.rag_chain.invoke(
        query=request.question,
        session_id=session_id,
        where=where,
    )

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceDocument(
                content=s.content[:500],
                source=s.source,
                relevance_score=round(s.score, 4),
                metadata=s.metadata,
            )
            for s in response.sources
        ],
        cited_sources=response.cited_sources,
        session_id=response.session_id,
        query_time_ms=response.query_time_ms,
        retrieval_time_ms=response.retrieval_time_ms,
        generation_time_ms=response.generation_time_ms,
        model_used=response.model_used,
    )
