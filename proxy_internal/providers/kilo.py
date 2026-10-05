"""Kilo Gateway provider — OpenAI-compatible API backed by Kilo Gateway."""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import build_oai_messages, _extract_oai_content
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

KILO_GATEWAY_BASE_URL = "https://api.kilo.ai/api/gateway"

_KILO_UNSUPPORTED_PARAMS = frozenset({"thinking", "reasoning", "top_k"})

def _resolve_kilo_model(model: str) -> str:
    """Strip the local Kilo provider prefix before calling the gateway."""
    if model.lower().startswith("kilo:"):
        return model[len("kilo:"):]
    return model


async def _fetch_kilo_models() -> list[dict]:
    """Fetch available Kilo models from the gateway model catalog."""
    import proxy

    auth_session = getattr(proxy.auth, "session", None)
    session = proxy.third_party_session or auth_session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not auth_session
    try:
        async with session.get(
            f"{KILO_GATEWAY_BASE_URL}/models",
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("data", [])
                if not models and isinstance(data, list):
                    models = data
                return [
                    {"id": f"kilo:{m['id']}", "object": "model", "owned_by": m.get("owned_by", "kilo")}
                    for m in models
                    if isinstance(m, dict) and m.get("id")
                ]
            logger.debug(f"Kilo /models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Kilo model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


async def call_kilo_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
) -> str:
    """Call the Kilo Gateway chat completions endpoint and return the full text."""
    import proxy

    if not proxy.kilo_api_key:
        raise HTTPException(status_code=503, detail="Kilo Gateway API key not configured")

    payload = {
        "model": _resolve_kilo_model(model),
        "messages": build_oai_messages(system_prompt, messages),
        "max_tokens": max_tokens,
    }
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in _KILO_UNSUPPORTED_PARAMS})

    headers = {
        "Authorization": f"Bearer {proxy.kilo_api_key}",
        "Content-Type": "application/json",
    }

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Kilo Gateway {model} ({len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(
            f"{KILO_GATEWAY_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Kilo Gateway {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise HTTPException(status_code=502, detail="Kilo Gateway returned no choices")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                raise HTTPException(status_code=502, detail="Kilo Gateway returned no content")
            return content

    try:
        text = await proxy._with_retry(
            _do_call,
            operation=f"Kilo Gateway {model}",
            request_id=request_id,
        )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Kilo Gateway)")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_kilo_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
):
    """Call the Kilo Gateway chat completions endpoint with streaming."""
    import proxy

    if not proxy.kilo_api_key:
        err, done = yield_sse_error(model, "[Kilo Gateway Error: API key not configured]")
        yield err
        yield done
        return

    payload = {
        "model": _resolve_kilo_model(model),
        "messages": build_oai_messages(system_prompt, messages),
        "max_tokens": max_tokens,
        "stream": True,
    }
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in _KILO_UNSUPPORTED_PARAMS})

    headers = {
        "Authorization": f"Bearer {proxy.kilo_api_key}",
        "Content-Type": "application/json",
    }

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Kilo Gateway {model} ({len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                f"{KILO_GATEWAY_BASE_URL}/chat/completions",
                json_payload=payload,
                headers=headers,
                operation="Kilo Gateway",
                request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[Kilo Gateway Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "Kilo Gateway", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] Kilo Gateway streaming error: {e}")
            err, done = yield_sse_error(model, f"[Kilo Gateway Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
