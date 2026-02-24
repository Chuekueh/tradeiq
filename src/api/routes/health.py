from fastapi import APIRouter

from src.api.dependencies import app_state
from src.api.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        vector_store_count=app_state.vector_store.count() if app_state.vector_store else 0,
        embedding_model=app_state.settings.embedding_model.value if app_state.settings else "",
        llm_model=app_state.settings.openai_model if app_state.settings else "",
    )
