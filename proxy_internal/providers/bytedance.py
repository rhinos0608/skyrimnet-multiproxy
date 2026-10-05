"""ByteDance ModelArk wrapper over the Anthropic-compatible adapter."""

import json
import logging
import uuid

import aiohttp
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from proxy_internal.anthropic_adapter import (
    call_anthropic_adapter_direct,
    call_anthropic_adapter_streaming,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

BYTEDANCE_ANTHROPIC_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/coding"


_BYTEDANCE_MODEL_ALIASES = {
    # Versioned API strings that the Coding Plan API rejects — use clean alias instead
    "kimi-k2-250905": "kimi-k2.5",
    "kimi-k2-250711": "kimi-k2.5",
    # Superseded GLM id on the Coding Plan — forward to the active GLM-5 line.
    "glm-4.7": "glm-5.1",
    # Dash-normalized variants the dashboard filter emits — restore dots/vendor prefix.
    "seed-2-0-pro": "dola-seed-2.0-pro",
    "seed-2-0-lite": "dola-seed-2.0-lite",
    "seed-2-0-code": "dola-seed-2.0-code",
}


# ByteDance has a max_tokens limit of 32768
BYTEDANCE_MAX_TOKENS = 32768


def _resolve_bytedance_model(model):
    """Strip the bytedance: prefix and apply alias overrides for upstream calls."""
    if model.lower().startswith("bytedance:"):
        model = model[len("bytedance:"):]
    return _BYTEDANCE_MODEL_ALIASES.get(model, model)


async def call_bytedance_direct(system_prompt, messages, model, max_tokens, **extra_params):
    import proxy

    if not proxy.bytedance_api_key:
        raise HTTPException(status_code=503, detail="ByteDance API key not configured")

    # Cap max_tokens to ByteDance's limit
    max_tokens = min(max_tokens, BYTEDANCE_MAX_TOKENS)

    return await call_anthropic_adapter_direct(
        system_prompt,
        messages,
        _resolve_bytedance_model(model),
        max_tokens,
        api_key=proxy.bytedance_api_key,
        base_url=BYTEDANCE_ANTHROPIC_BASE_URL,
        session=proxy.third_party_session,
        **extra_params,
    )


async def call_bytedance_streaming(system_prompt, messages, model, max_tokens, **extra_params):
    import proxy

    if not proxy.bytedance_api_key:
        error, done = yield_sse_error(
            model, "[ByteDance Error: API key not configured]", status_code=503
        )
        yield error
        yield done
        return

    # Cap max_tokens to ByteDance's limit
    max_tokens = min(max_tokens, BYTEDANCE_MAX_TOKENS)

    async for chunk in call_anthropic_adapter_streaming(
        system_prompt,
        messages,
        _resolve_bytedance_model(model),
        max_tokens,
        api_key=proxy.bytedance_api_key,
        base_url=BYTEDANCE_ANTHROPIC_BASE_URL,
        session=proxy.third_party_session,
        **extra_params,
    ):
        yield chunk


async def call_bytedance_messages_passthrough(body: dict, stream: bool):
    """Forward an Anthropic-shaped /v1/messages body directly to ByteDance's
    Anthropic-compatible endpoint. Returns a StreamingResponse (stream=True)
    or parsed JSON dict (stream=False). Used by /v1/messages when tools are
    present so tool_use/tool_result blocks survive end-to-end.
    """
    import proxy
    import time

    if not proxy.bytedance_api_key:
        raise HTTPException(status_code=503, detail="ByteDance API key not configured")

    patched = dict(body)
    patched["model"] = _resolve_bytedance_model(patched.get("model", ""))
    patched["stream"] = bool(stream)

    # Cap max_tokens to ByteDance's limit
    if patched.get("max_tokens", 0) > BYTEDANCE_MAX_TOKENS:
        patched["max_tokens"] = BYTEDANCE_MAX_TOKENS
        logger.debug(f"Capped max_tokens to {BYTEDANCE_MAX_TOKENS} for ByteDance")

    endpoint = f"{BYTEDANCE_ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "x-api-key": proxy.bytedance_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    request_id = uuid.uuid4().hex[:8]
    model_name = patched.get("model", "?")
    msg_count = len(patched.get("messages") or [])
    logger.info(
        f"[{request_id}] -> {model_name} (bytedance /v1/messages passthrough, "
        f"{msg_count} msgs, stream={stream})"
    )
    start = time.time()

    shared = proxy.third_party_session
    if shared is not None and not shared.closed:
        session = shared
    else:
        session = proxy.create_session()
    owns_session = session is not shared
    try:
        if not stream:
            try:
                async with session.post(endpoint, headers=headers, json=patched) as resp:
                    data = await resp.json()
                    elapsed = time.time() - start
                    if resp.status >= 400:
                        logger.error(f"[{request_id}] bytedance passthrough {resp.status}: {str(data)[:300]}")
                        raise HTTPException(status_code=resp.status, detail=str(data)[:500])
                    logger.info(f"[{request_id}] <- bytedance passthrough done ({elapsed:.1f}s)")
                    return data
            finally:
                if owns_session:
                    await session.close()

        async def _stream_body():
            try:
                async with session.post(endpoint, headers=headers, json=patched) as resp:
                    if resp.status >= 400:
                        err = await resp.text()
                        logger.error(f"[{request_id}] bytedance passthrough stream {resp.status}: {err[:300]}")
                        yield f"data: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': err[:200]}})}\n\n"
                        return
                    async for chunk in resp.content.iter_any():
                        yield chunk
            finally:
                if owns_session:
                    await session.close()

        return StreamingResponse(
            _stream_body(),
            media_type="text/event-stream",
        )
    except HTTPException:
        # Session already closed by inner try/finally or _stream_body() generator.
        raise
    except Exception as exc:
        # Session already closed by inner try/finally or _stream_body() generator.
        raise HTTPException(status_code=502, detail=f"ByteDance passthrough failed: {exc}") from exc
