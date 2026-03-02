"""
API Routes for SLAgent

NDJSON streaming endpoints for the RAG chat interface.
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request schema"""
    query: str
    request_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response schema (for non-streaming)"""
    answer: str
    references: list


def get_orchestrator(request: Request):
    """
    Get orchestrator instance from app state.

    Args:
        request: FastAPI request object

    Returns:
        Orchestrator instance

    Raises:
        HTTPException: If orchestrator not initialized
    """
    app_state = getattr(request.app, 'state', None)
    if app_state is None:
        # Fallback to global app_state dict (from main.py)
        from main import app_state as global_app_state
        orchestrator = global_app_state.get('orchestrator')
    else:
        orchestrator = getattr(app_state, 'orchestrator', None)

    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized. Service starting up."
        )

    return orchestrator


@router.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """
    Streaming chat endpoint.

    Returns NDJSON stream with event types:
    - progress: {"type": "progress", "data": "message"}
    - references: {"type": "references", "data": [...]}
    - thinking: {"type": "thinking", "data": "thought"}
    - token: {"type": "token", "data": "word"}
    - error: {"type": "error", "data": "error message"}
    """
    # Get orchestrator from app state
    orchestrator = get_orchestrator(req)

    async def event_generator():
        """Generate NDJSON events from orchestrator"""
        try:
            async for event in orchestrator.process_query(
                request.query,
                request_id=request.request_id
            ):
                yield json.dumps(event) + "\n"

        except Exception as e:
            logger.error("Chat endpoint error", error=str(e), exc_info=True)
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson"
    )


@router.post("/chat/sync")
async def chat_sync(request: ChatRequest, req: Request):
    """
    Non-streaming chat endpoint (for testing/debugging).

    Collects all tokens and returns complete response.
    """
    orchestrator = get_orchestrator(req)

    tokens = []
    references = []

    try:
        async for event in orchestrator.process_query(
            request.query,
            request_id=request.request_id
        ):
            if event["type"] == "token":
                tokens.append(event["data"])
            elif event["type"] == "references":
                references = event["data"]
            elif event["type"] == "error":
                raise HTTPException(status_code=500, detail=event["data"])

        return ChatResponse(
            answer="".join(tokens),
            references=references
        )

    except Exception as e:
        logger.error("Sync chat error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
