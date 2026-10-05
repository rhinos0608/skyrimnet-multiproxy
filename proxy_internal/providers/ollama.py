"""Ollama OpenAI-compatible provider (Cloud + local daemon)."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    _has_reasoning_without_content,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


_OLLAMA_UNSUPPORTED_PARAMS = {"top_k", "thinking", "reasoning"}


def _ollama_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Normalize reasoning controls for Ollama's OpenAI-compatible endpoint."""
    reasoning_effort = extra_params.get("reasoning_effort")
    reasoning = extra_params.get("reasoning")
    thinking = extra_params.get("thinking")

    if reasoning_effort is None and isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            reasoning_effort = "none"
        elif reasoning.get("effort") is not None:
            reasoning_effort = reasoning["effort"]
    elif reasoning_effort is None and reasoning is False:
        reasoning_effort = "none"

    if reasoning_effort is None and isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            reasoning_effort = "none"
    elif reasoning_effort is None and thinking is False:
        reasoning_effort = "none"

    payload.update({
        k: v for k, v in extra_params.items()
        if v is not None and k not in _OLLAMA_UNSUPPORTED_PARAMS
    })
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort


def _resolve_ollama_model(model: str) -> str:
    """Strip the local provider prefix (case-insensitive)."""
    if model.lower().startswith("ollama:"):
        return model[len("ollama:"):]
    return model


async def _fetch_ollama_models(
    local_url: str = "http://localhost:11434/v1/models",
    cloud_url: str = "https://ollama.com/v1/models",
    api_key: str | None = None,
    timeout_s: float = 5.0,
) -> list[dict]:
    """Provider-owned discovery: local daemon plus cloud (when key set).

    Returns `ollama:<id>` entries; empty when daemon offline and no key.
    Shared endpoints can call this instead of probing only `ollama list`.
    """
    import proxy as _proxy

    key = api_key if api_key is not None else getattr(_proxy, "ollama_api_key", None)
    urls = [local_url]
    if key:
        urls.append(cloud_url)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    seen: set[str] = set()
    out: list[dict] = []
    session = getattr(_proxy, "ollama_session", None) or getattr(getattr(_proxy, "auth", None), "session", None)
    try:
        import aiohttp as _aiohttp

        close = False
        if session is None or session.closed:
            session = _aiohttp.ClientSession()
            close = True
        try:
            for url in urls:
                try:
                    async with session.get(
                        url, headers=headers,
                        timeout=_aiohttp.ClientTimeout(total=timeout_s),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        items = data.get("data", []) if isinstance(data, dict) else data
                        for m in items if isinstance(items, list) else []:
                            mid = m.get("id") if isinstance(m, dict) else None
                            if mid and mid not in seen:
                                seen.add(mid)
                                out.append({"id": f"ollama:{mid}", "object": "model", "owned_by": "ollama"})
                except Exception:
                    continue
        finally:
            if close:
                await session.close()
    except Exception:
        return out
    return out


async def call_ollama_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Ollama (OpenAI-compatible), collect full response."""
    import proxy
    api_model = _resolve_ollama_model(model)
    if proxy.ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {proxy.ollama_api_key}", "Content-Type": "application/json"}
        # Ollama Cloud models have output limits ranging from 32K-262K tokens.
        # Cap max_tokens at 32768 (the lowest max output across all cloud models)
        # to avoid exceeding per-model caps which can cause server-side truncation.
        max_tokens = min(max_tokens, 32768)
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if proxy.ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs)")
    start = time.time()

    session = proxy.ollama_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.ollama_session and not proxy.auth.session

    async def _do_call() -> str:
        # ClientConnectorError means the local Ollama daemon is down — short
        # circuit so _with_retry doesn't waste attempts on a dead socket.
        try:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status in (401, 403):
                    raise HTTPException(status_code=401, detail="Ollama Cloud auth failed — check API key")
                if resp.status == 429:
                    raise HTTPException(status_code=429, detail="Ollama rate limit exceeded")
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise HTTPException(status_code=502, detail="Ollama returned no choices")
                choice = choices[0]
                text = _extract_oai_content(choice.get("message", {}))
                if text is None:
                    if (
                        choice.get("finish_reason") == "stop"
                        and not _has_reasoning_without_content(data)
                    ):
                        return ""
                    raise HTTPException(status_code=502, detail="Ollama returned no content")
                return text
        except aiohttp.ClientConnectorError:
            # Local daemon down — not transient, surface immediately without retry.
            raise HTTPException(status_code=503, detail="Ollama not running at localhost:11434")

    try:
        # Ollama Cloud blips are transient — use longer backoff and more retries
        # than the global default (which is tuned for local/fast providers).
        retry_kwargs = (
            {"base_delay_s": 2.0, "max_retries_override": 2}
            if proxy.ollama_api_key
            else {"retry_on_status": frozenset({429, 500, 502, 504})}
        )
        text = await proxy._with_retry(
            _do_call,
            operation=f"Ollama {api_model}",
            request_id=request_id,
            **retry_kwargs,
        )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Ollama)")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_ollama_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Ollama with streaming, passthrough SSE directly.

    Retry policy: pre-first-byte retries via ``_open_stream_with_retry``.  A
    ``ClientConnectorError`` (local daemon down) is NOT retried — surfaced as
    an SSE error chunk immediately.
    """
    import proxy
    api_model = _resolve_ollama_model(model)
    if proxy.ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {proxy.ollama_api_key}", "Content-Type": "application/json"}
        # Ollama Cloud models have output limits ranging from 32K-262K tokens.
        # Cap max_tokens at 32768 (the lowest max output across all cloud models)
        # to avoid exceeding per-model caps which can cause server-side truncation.
        max_tokens = min(max_tokens, 32768)
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if proxy.ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.ollama_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.ollama_session and not proxy.auth.session
    resp_cm = None
    try:
        cloud_retry_kwargs = (
            {"base_delay_s": 2.0, "max_retries_override": 2}
            if proxy.ollama_api_key
            else {"retry_on_status": frozenset({429, 500, 502, 504})}
        )
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                endpoint,
                json_payload=payload,
                headers=headers,
                operation="Ollama",
                request_id=request_id,
                retry_connect_errors=False,
                **cloud_retry_kwargs,
            )
        except aiohttp.ClientConnectorError:
            err, done = yield_sse_error(model, "[Ollama Error: not running at localhost:11434]")
            yield err; yield done
            return
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[Ollama Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "Ollama", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                # Ollama Cloud frequently drops connections mid-stream
                # ("unexpected EOF" / cloud proxy failure) without sending
                # the [DONE] terminator. Emit a synthetic [DONE] so the
                # client (SkyrimNet) doesn't hang waiting for completion.
                logger.warning(
                    f"[{request_id}] Ollama stream ended without [DONE] — "
                    f"connection dropped mid-generation (cloud proxy failure)"
                )
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] Ollama streaming error: {e}")
            err, done = yield_sse_error(model, f"[Ollama Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
