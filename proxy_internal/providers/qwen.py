"""Qwen Code OpenAI-compatible provider (portal.qwen.ai)."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    _is_reasoning_truncated,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

# Legacy Qwen Portal alias -> modern catalog id. `coder-model` is an opaque
# alias from early CLI wrappers, absent from the modern portal catalog.
# Resolve it to the official portal default so old configs keep working.
_QWEN_MODEL_ALIASES = {
    "coder-model": "qwen3.5-plus",
}


def _resolve_qwen_model(model: str) -> str:
    """Strip the qwen: prefix and resolve legacy aliases to modern ids."""
    api_model = model[len("qwen:"):] if model.lower().startswith("qwen:") else model
    return _QWEN_MODEL_ALIASES.get(api_model, _QWEN_MODEL_ALIASES.get(api_model.lower(), api_model))

_reasoning_strip_warned = False


async def call_qwen_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Qwen (OpenAI-compatible), collect full response."""
    import proxy
    # Re-read credentials file in case the CLI refreshed the token.
    if not await proxy.qwen_auth.refresh_if_needed():
        raise HTTPException(status_code=503, detail="Qwen auth not ready -- run Qwen Code CLI login first")

    api_model = _resolve_qwen_model(model)
    endpoint = f"{proxy.QWEN_BASE_URL}/chat/completions"

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for Qwen "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Qwen ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.qwen_auth.session or proxy.create_session()
    owns_session = not proxy.qwen_auth.session

    async def _do_call() -> str:
        headers = proxy.qwen_auth.get_auth_headers()
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status in (400, 401, 403):
                # Token may be stale — try reloading from file and retry once
                # inline (this auth-recovery path must not be conflated with the
                # 401/403 fail-fast rule in _with_retry).
                if proxy.qwen_auth.reload_from_file():
                    retry_headers = proxy.qwen_auth.get_auth_headers()
                    async with session.post(endpoint, json=payload, headers=retry_headers) as retry_resp:
                        if retry_resp.status == 200:
                            data = await retry_resp.json()
                            content = _extract_oai_content(data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                return content
                error_body = await resp.text()
                # 401/403/400 on final failure — _with_retry will fail-fast on
                # 401/403, and a 400 is not retryable either.
                raise HTTPException(status_code=resp.status, detail=f"Qwen error — {error_body[:200]}. Run Qwen Code CLI to refresh auth.")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Qwen rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Qwen {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="Qwen returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] Qwen reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    retry_headers = proxy.qwen_auth.get_auth_headers()
                    async with session.post(endpoint, json=retry_payload, headers=retry_headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(
                                retry_data.get("choices", [{}])[0].get("message", {})
                            )
                            if content:
                                return content
                raise HTTPException(status_code=500, detail="Qwen returned empty content")
            return content

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"Qwen {api_model}", request_id=request_id)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Qwen request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Qwen)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_qwen_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Qwen with streaming, passthrough SSE directly."""
    import proxy
    if not await proxy.qwen_auth.refresh_if_needed():
        err, done = yield_sse_error(model, "[Qwen Error: auth not ready — run Qwen Code CLI login]")
        yield err; yield done
        return

    api_model = _resolve_qwen_model(model)
    endpoint = f"{proxy.QWEN_BASE_URL}/chat/completions"
    headers = proxy.qwen_auth.get_auth_headers()

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True, "max_tokens": max_tokens}
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for Qwen "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Qwen ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.qwen_auth.session or proxy.create_session()
    owns_session = not proxy.qwen_auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session, endpoint,
                json_payload=payload, headers=headers,
                operation="Qwen", request_id=request_id,
            )
        except HTTPException as e:
            # 400/401/403 — try a one-shot auth reload + retry before giving up.
            if e.status_code in (400, 401, 403) and proxy.qwen_auth.reload_from_file():
                retry_headers = proxy.qwen_auth.get_auth_headers()
                try:
                    async with session.post(endpoint, json=payload, headers=retry_headers) as retry_resp:
                        if retry_resp.status == 200:
                            saw_done = False
                            async for event in proxy.passthrough_sse(retry_resp, request_id, "Qwen", start):
                                if event.strip().startswith("data: [DONE]"):
                                    saw_done = True
                                yield event
                            if not saw_done:
                                yield "data: [DONE]\n\n"
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    pass
                err, done = yield_sse_error(model, "[Qwen Error: auth failed]")
                yield err; yield done
                return
            err, done = yield_sse_error(model, f"[Qwen Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "Qwen", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] Qwen streaming error: {e}")
            err, done = yield_sse_error(model, f"[Qwen Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
