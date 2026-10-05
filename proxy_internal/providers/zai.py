"""Z.AI OpenAI-compatible provider."""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.context_trim import (
    MAX_TRIM_ROUNDS,
    ContextExceededError,
    context_excess_tokens,
    trim_messages_for_context,
)
from proxy_internal.message_normalize import (
    _extract_oai_content,
    _is_reasoning_truncated,
    build_oai_messages,
)
from proxy_internal.sse_utils import make_sse_content_chunk, yield_sse_error

logger = logging.getLogger("proxy")

# Legacy GLM-4 ids -> modern Coding Plan ids. GLM-4.5/4.6 are deprecated;
# resolve stale pins to the active GLM-5 series so old configs keep working.
_ZAI_MODEL_ALIASES = {
    "glm-4.5-air": "glm-5.3",
    "glm-4.5": "glm-5.2",
    "glm-4.6": "glm-5-turbo",
}


def _resolve_zai_model(model: str) -> str:
    """Strip the zai: prefix and resolve legacy ids to modern replacements."""
    api_model = model[len("zai:"):] if model.lower().startswith("zai:") else model
    return _ZAI_MODEL_ALIASES.get(api_model, _ZAI_MODEL_ALIASES.get(api_model.lower(), api_model))

_reasoning_strip_warned = False


def _zai_supports_vision(api_model: str) -> bool:
    """Z.AI vision models contain 'v' as a suffix/segment (e.g. glm-5v-turbo, glm-4.6v)."""
    low = api_model.lower()
    # Match patterns like "glm-4.6v", "glm-5v-turbo", "glm-4.1v-thinking-flash"
    for part in low.replace(".", "-").split("-"):
        if part.endswith("v") and len(part) > 1:
            return True
    # Also match dedicated vision model names
    if "ocr" in low or "autoglm" in low:
        return True
    return False


def _raise_if_context_exceeded(status: int, body: str) -> None:
    """Abort a stream request Z.AI rejected for exceeding the context window.

    Passed to ``_open_stream_with_retry`` as ``error_body_hook`` so the caller
    can trim history and retry instead of retrying a payload that cannot fit.
    """
    excess = context_excess_tokens(status, body)
    if excess is not None:
        raise ContextExceededError(excess)


async def call_zai_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Z.AI (OpenAI-compatible), collect full response.

    A rejection for exceeding the model's context window drops the oldest
    droppable history and retries, so a long conversation degrades instead of
    failing outright.
    """
    import proxy
    if not proxy.zai_api_key:
        raise HTTPException(status_code=503, detail="Z.AI API key not configured — add it via /config/zai-key")
    api_model = _resolve_zai_model(model)
    endpoint = f"{proxy.ZAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.zai_api_key}", "Content-Type": "application/json"}
    strip = not _zai_supports_vision(api_model)

    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for Z.AI "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Z.AI ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    # Rebuilt per attempt so a context-trim retry sends the shortened history.
    current_messages: list = messages

    def _build_payload() -> dict:
        payload = {
            "model": api_model,
            "messages": build_oai_messages(system_prompt, current_messages, strip_vision=strip),
            "max_tokens": max_tokens,
        }
        payload.update({k: v for k, v in extra_params.items()
                        if v is not None and k not in ("thinking", "reasoning")})
        return payload

    async def _do_call() -> str:
        async with session.post(endpoint, json=_build_payload(), headers=headers) as resp:
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Z.AI auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Z.AI rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                excess = context_excess_tokens(resp.status, error_body)
                if excess is not None:
                    raise ContextExceededError(excess)
                logger.error(f"[{request_id}] Z.AI {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"[{request_id}] Z.AI returned empty choices array")
                raise HTTPException(status_code=500, detail="Z.AI returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                # Reasoning model may have exhausted max_tokens on chain-of-thought — retry without cap
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] Z.AI reasoning truncated, retrying without max_tokens")
                    retry_payload = _build_payload()
                    retry_payload.pop("max_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                return content
                logger.warning(f"[{request_id}] Z.AI returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="Z.AI returned empty content")
            return content

    try:
        for _round in range(MAX_TRIM_ROUNDS + 1):
            try:
                content = await proxy._with_retry(_do_call, operation=f"Z.AI {api_model}", request_id=request_id)
                break
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Z.AI request timed out")
            except ContextExceededError as exc:
                current_messages, removed = trim_messages_for_context(
                    current_messages, exc.excess_tokens
                )
                if removed == 0:
                    logger.error(
                        f"[{request_id}] Z.AI rejected the prompt for exceeding the context "
                        f"window by {exc.excess_tokens} tokens and no droppable history remains"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Z.AI rejected the prompt as too long for the model's context "
                               "window, and there is no older history left to drop.",
                    )
                logger.warning(
                    f"[{request_id}] Z.AI context window exceeded by {exc.excess_tokens} tokens "
                    f"— dropped {removed:,} chars of older history, retrying"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Z.AI rejected the prompt as too long for the model's context window "
                       "even after dropping older history.",
            )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Z.AI)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_zai_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Z.AI with streaming, passthrough SSE directly.

    A rejection for exceeding the model's context window drops the oldest
    droppable history and retries. That retry is safe to replay to the client
    because Z.AI reports context exhaustion while opening the stream, before any
    event has been yielded.
    """
    import proxy
    if not proxy.zai_api_key:
        err, done = yield_sse_error(model, "[Z.AI Error: API key not configured]")
        yield err; yield done
        return

    request_id = uuid.uuid4().hex[:8]
    current_messages: list = messages
    for _round in range(MAX_TRIM_ROUNDS + 1):
        try:
            async for event in _zai_stream_attempt(
                system_prompt, current_messages, model, max_tokens, request_id, **extra_params
            ):
                yield event
            return
        except ContextExceededError as exc:
            current_messages, removed = trim_messages_for_context(
                current_messages, exc.excess_tokens
            )
            if removed == 0:
                logger.error(
                    f"[{request_id}] Z.AI rejected the prompt for exceeding the context "
                    f"window by {exc.excess_tokens} tokens and no droppable history remains"
                )
                err, done = yield_sse_error(
                    model,
                    "[Z.AI Error 400: prompt too long for the model's context window, "
                    "and no older history left to drop]",
                )
                yield err; yield done
                return
            logger.warning(
                f"[{request_id}] Z.AI context window exceeded by {exc.excess_tokens} tokens "
                f"— dropped {removed:,} chars of older history, retrying"
            )

    err, done = yield_sse_error(
        model,
        "[Z.AI Error 400: prompt still too long for the model's context window "
        "after dropping older history]",
    )
    yield err; yield done


async def _zai_stream_attempt(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    request_id: str,
    **extra_params,
):
    """One streaming attempt; raises ContextExceededError before yielding if too long."""
    import proxy
    api_model = _resolve_zai_model(model)
    endpoint = f"{proxy.ZAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.zai_api_key}", "Content-Type": "application/json"}
    strip = not _zai_supports_vision(api_model)

    # Omit max_tokens for streaming — reasoning models (GLM) exhaust it on chain-of-thought
    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages, strip_vision=strip), "stream": True}
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for Z.AI "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})

    logger.info(f"[{request_id}] -> Z.AI ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    try:
        # Z.AI sometimes returns a 200 with an empty streaming body; retry a couple times.
        # (This outer loop is the empty-body semantic retry.  Network-level
        # retries on the initial POST are handled per-attempt by
        # _open_stream_with_retry below.)
        for attempt in range(3):
            resp_cm = None
            try:
                resp, resp_cm = await proxy._open_stream_with_retry(
                    session, endpoint,
                    json_payload=payload, headers=headers,
                    operation="Z.AI", request_id=request_id,
                    error_body_hook=_raise_if_context_exceeded,
                )
            except HTTPException as e:
                err, done = yield_sse_error(model, f"[Z.AI Error {e.status_code}]")
                yield err; yield done
                return
            try:
                headers_obj = getattr(resp, "headers", None)
                content_type_val = ""
                if headers_obj is not None:
                    try:
                        content_type_val = headers_obj.get("Content-Type") or ""
                        if asyncio.iscoroutine(content_type_val):
                            content_type_val = await content_type_val
                    except Exception:
                        content_type_val = ""
                content_type = str(content_type_val).lower()

                # Fallback: if upstream responds with non-streaming JSON despite stream=True.
                if "application/json" in content_type:
                    data = await resp.json()
                    content = _extract_oai_content(
                        data.get("choices", [{}])[0]
                        .get("message", {})
                    )
                    if not content:
                        err, done = yield_sse_error(model, "[Z.AI Error: empty JSON response]")
                        yield err; yield done
                        return
                    yield make_sse_content_chunk(model, content)
                    yield "data: [DONE]\n\n"
                    return

                yielded_any = False
                saw_done = False
                total_content = 0
                async for event in proxy.passthrough_sse(resp, request_id, "Z.AI", start):
                    yielded_any = True
                    if event.strip().startswith("data: [DONE]"):
                        saw_done = True
                    elif event.startswith("data: ") and '"content"' in event:
                        try:
                            d = json.loads(event[6:])
                            c = d.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                            total_content += len(c)
                        except Exception:
                            pass
                    yield event

                if not yielded_any or total_content == 0:
                    logger.warning(
                        f"[{request_id}] Z.AI returned empty stream body (attempt {attempt + 1}/3)"
                        if not yielded_any else
                        f"[{request_id}] Z.AI stream had no content (attempt {attempt + 1}/3)"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue

                    # Last resort: try a non-streaming request and convert to SSE.
                    try:
                        direct_payload = dict(payload)
                        direct_payload.pop("stream", None)
                        async with session.post(endpoint, json=direct_payload, headers=headers) as direct_resp:
                            if direct_resp.status != 200:
                                err, done = yield_sse_error(
                                    model,
                                    f"[Z.AI Error {direct_resp.status}: empty stream + direct fallback failed]",
                                )
                                yield err; yield done
                                return
                            data = await direct_resp.json()
                            content = _extract_oai_content(
                                data.get("choices", [{}])[0]
                                .get("message", {})
                            )
                            if content:
                                yield make_sse_content_chunk(model, content)
                                yield "data: [DONE]\n\n"
                                return
                            else:
                                logger.warning(f"[{request_id}] Z.AI direct fallback also returned empty content")
                    except Exception as e:
                        logger.error(f"[{request_id}] Z.AI direct fallback failed after empty stream: {e}")

                    err, done = yield_sse_error(model, "[Z.AI Error: empty stream response]")
                    yield err; yield done
                    return

                if not saw_done:
                    yield "data: [DONE]\n\n"
                return
            finally:
                if resp_cm is not None:
                    try:
                        await resp_cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    resp_cm = None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Z.AI streaming error: {e}")
        err, done = yield_sse_error(model, f"[Z.AI Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()
