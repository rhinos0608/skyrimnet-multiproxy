"""LM Studio OpenAI-compatible provider (local only)."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

_LMSTUDIO_UNSUPPORTED_PARAMS = {"top_k", "thinking", "reasoning", "reasoning_effort"}


def _lmstudio_payload_fixup(payload: dict, extra_params: dict) -> None:
    payload.update({
        k: v for k, v in extra_params.items()
        if v is not None and k not in _LMSTUDIO_UNSUPPORTED_PARAMS
    })


def _resolve_lmstudio_model(model: str) -> str:
    """Strip the local provider prefix (case-insensitive)."""
    if model.lower().startswith("lmstudio:"):
        return model[len("lmstudio:"):]
    return model


def _lmstudio_base_url(port: int | None = None) -> str:
    """Base URL for LM Studio server (default port 1234)."""
    import proxy as _proxy

    p = port if port is not None else getattr(_proxy, "lmstudio_port", 1234) or 1234
    return f"http://localhost:{p}/v1"


async def _fetch_lmstudio_models(base_url: str | None = None, timeout_s: float = 5.0) -> list[dict]:
    """Provider-owned discovery: probe `<base>/models` for loaded models.

    Returns `lmstudio:<id>` entries; empty when server offline.
    Shared endpoints can call this (mirrors /config/ollama-models pattern).
    """
    import proxy as _proxy

    url = f"{(base_url or _lmstudio_base_url()).rstrip('/')}/models"
    out: list[dict] = []
    try:
        import aiohttp as _aiohttp

        session = getattr(_proxy, "lmstudio_session", None) or getattr(getattr(_proxy, "auth", None), "session", None)
        close = False
        if session is None or session.closed:
            session = _aiohttp.ClientSession()
            close = True
        try:
            async with session.get(url, timeout=_aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                items = data.get("data", []) if isinstance(data, dict) else data
                for m in items if isinstance(items, list) else []:
                    mid = m.get("id") if isinstance(m, dict) else None
                    if mid:
                        out.append({"id": f"lmstudio:{mid}", "object": "model", "owned_by": "lmstudio"})
        finally:
            if close:
                await session.close()
    except Exception:
        return []
    return out


async def call_lmstudio_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to LM Studio (OpenAI-compatible), collect full response."""
    import proxy
    api_model = _resolve_lmstudio_model(model)
    endpoint = f"{_lmstudio_base_url()}/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _lmstudio_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> LM Studio local ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.lmstudio_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.lmstudio_session and not proxy.auth.session

    async def _do_call() -> str:
        try:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status == 429:
                    raise HTTPException(status_code=429, detail="LM Studio rate limit exceeded")
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] LM Studio {resp.status}: {error_body[:300]}")
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise HTTPException(status_code=502, detail="LM Studio returned no choices")
                text = _extract_oai_content(choices[0].get("message", {}))
                if text is None:
                    raise HTTPException(status_code=502, detail="LM Studio returned no content")
                return text
        except aiohttp.ClientConnectorError:
            raise HTTPException(status_code=503, detail="LM Studio not running at localhost:1234")

    try:
        text = await proxy._with_retry(
            _do_call,
            operation=f"LM Studio {api_model}",
            request_id=request_id,
            retry_on_status=frozenset({429, 500, 502, 504}),
        )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, LM Studio)")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_lmstudio_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to LM Studio with streaming, passthrough SSE directly."""
    import proxy
    api_model = _resolve_lmstudio_model(model)
    endpoint = f"{_lmstudio_base_url()}/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    _lmstudio_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> LM Studio local ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.lmstudio_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.lmstudio_session and not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                endpoint,
                json_payload=payload,
                headers=headers,
                operation="LM Studio",
                request_id=request_id,
                retry_connect_errors=False,
                retry_on_status=frozenset({429, 500, 502, 504}),
            )
        except aiohttp.ClientConnectorError:
            err, done = yield_sse_error(model, "[LM Studio Error: not running at localhost:1234]")
            yield err; yield done
            return
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[LM Studio Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "LM Studio", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] LM Studio streaming error: {e}")
            err, done = yield_sse_error(model, f"[LM Studio Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
