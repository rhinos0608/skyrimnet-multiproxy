"""SSE chunk builders for OpenAI-compatible and Anthropic streaming responses."""

import json
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from contextvars import ContextVar
from typing import Optional

from fastapi import HTTPException

from proxy_internal._json import json_dumps


_stream_preflight_active: ContextVar[bool] = ContextVar(
    "stream_preflight_active", default=False
)
_ERROR_STATUS_RE = re.compile(r"\b[Ee]rror\s+([45]\d\d)\b")


def make_sse_error_chunk(model: str, error_msg: str) -> str:
    """Build a single SSE error chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": error_msg}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def make_sse_content_chunk(model: str, content: str) -> str:
    """Build a single SSE content chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def yield_sse_error(
    model: str, error_msg: str, *, status_code: Optional[int] = None
):
    """Return SSE error chunks, or raise HTTPException during active stream preflight.

    When ``_stream_preflight_active`` is set, this function raises ``HTTPException``
    instead of returning SSE chunks. Callers must invoke it outside broad
    ``try/except Exception`` handlers that would swallow the HTTPException.
    When preflight is inactive, returns ``(error_chunk, done_chunk)`` as SSE strings.
    """
    if _stream_preflight_active.get():
        match = _ERROR_STATUS_RE.search(error_msg)
        inferred_status = int(match.group(1)) if match else 502
        safe_status = (
            status_code
            if status_code is not None and 400 <= status_code <= 599
            else inferred_status
        )
        raise HTTPException(status_code=safe_status, detail=error_msg)
    return make_sse_error_chunk(model, error_msg), "data: [DONE]\n\n"


async def _close_async_iterator(iterator) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        with suppress(Exception):
            await close()


async def preflight_sse_stream(
    events: AsyncIterable[str | bytes],
) -> AsyncIterator[str | bytes]:
    """Advance a stream before response start, then replay its first item once."""
    iterator = events.__aiter__()
    token = _stream_preflight_active.set(True)
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        async def _empty():
            return
            yield  # pragma: no cover — makes this an async generator

        return _empty()
    except BaseException:
        await _close_async_iterator(iterator)
        raise
    finally:
        _stream_preflight_active.reset(token)

    async def _replay():
        try:
            yield first
            async for event in iterator:
                yield event
        finally:
            await _close_async_iterator(iterator)

    return _replay()


def _format_anthropic_sse(event: str, data: dict) -> bytes:
    """Format an Anthropic SSE event (named event + JSON data line).

    Returns ``bytes`` directly rather than a ``str``.  ``json_dumps`` is
    orjson-backed when available and always yields bytes; previously we
    decoded those bytes into a UTF-8 string just so ``StreamingResponse``
    could re-encode them back to bytes on the way out to the client.  In
    the per-token streaming hot path that decode + re-encode cycle is pure
    overhead — FastAPI's ``StreamingResponse`` accepts both ``str`` and
    ``bytes`` chunks, so callers don't need to change.
    """
    return b"event: " + event.encode("ascii") + b"\ndata: " + json_dumps(data) + b"\n\n"
