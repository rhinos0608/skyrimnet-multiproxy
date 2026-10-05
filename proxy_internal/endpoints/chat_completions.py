"""OpenAI-compatible chat completions endpoint."""
import logging

import aiohttp
from fastapi import APIRouter, HTTPException, Request

from proxy_internal.schemas import ChatRequest

router = APIRouter()
logger = logging.getLogger("proxy")


@router.post("/v1/chat/completions")
@router.post("/api/chat/completions")
@router.post("/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    import proxy
    try:
        return await proxy._chat_completions_inner(
            req, opencode_session_id=request.headers.get("x-opencode-session"),
        )
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
            aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
        logger.debug("Client disconnected mid-request — suppressing traceback")
        raise HTTPException(status_code=499, detail="Client disconnected")
