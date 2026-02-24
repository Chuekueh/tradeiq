from fastapi import APIRouter, HTTPException

from src.api.dependencies import app_state
from src.api.schemas.response import ConversationListResponse, ConversationSession

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations() -> ConversationListResponse:
    if not app_state.memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    sessions = app_state.memory.get_sessions()
    return ConversationListResponse(
        sessions=[
            ConversationSession(
                session_id=str(s["session_id"]),
                message_count=int(s["message_count"]),
                last_active=float(s["last_active"]),
            )
            for s in sessions
        ]
    )


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str) -> dict:
    if not app_state.memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    history = app_state.memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str) -> dict:
    if not app_state.memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    app_state.memory.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}
