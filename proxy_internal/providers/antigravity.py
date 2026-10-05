"""Antigravity (Google IDE) provider with multi-account OAuth fallback."""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import HTTPException

from proxy_internal.sse_parser import IncrementalSSEDecoder, parse_sse_frame
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


_ANTIGRAVITY_RETRYABLE_STATUSES = {403, 429, 502, 503, 504}


def ensure_account_session(account):
    """Create aiohttp session for account if missing.

    Synchronous helper — no await, safe on single event loop.
    Re-check after any await that may have yielded to another coroutine.
    """
    import proxy
    if not account.session:
        account.session = proxy.create_session()


def _convert_messages_to_antigravity(messages: list, system_prompt: Optional[str] = None) -> dict:
    """Convert OpenAI-format messages to Antigravity/Gemini-style request format."""
    # Build contents array (Gemini-style)
    contents = []
    for m in messages:
        role = m["role"]
        # Map assistant -> model for Gemini format
        gemini_role = "model" if role == "assistant" else role
        contents.append({
            "role": gemini_role,
            "parts": [{"text": m["content"]}]
        })

    request_body = {
        "contents": contents,
    }

    # System instruction must be an object with parts
    if system_prompt:
        request_body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    return request_body


def _get_antigravity_model_id(model: str) -> str:
    """Convert proxy model name to Antigravity API model ID.

    External names KEEP the ``antigravity-`` routing prefix so routes stay
    distinguishable from normal Claude/Gemini traffic; the prefix is stripped
    (exactly once) here at the Antigravity provider boundary because the
    upstream catalog uses bare runtime IDs (e.g. ``claude-sonnet-4-6``,
    ``claude-opus-4-6-thinking``). Tier aliases follow the installed
    pi-provider-antigravity reference: bare base + default high tier, and
    catalog ``deprecatedModelIds`` renames (``gemini-3.1-pro-high`` ->
    ``gemini-pro-agent``) applied before send.
    """
    model_lower = model.lower()

    # Strip exactly one routing prefix; upstream IDs are bare in all families.
    if model_lower.startswith("antigravity-"):
        model_lower = model_lower[len("antigravity-"):]

    # Map model names to Antigravity API model IDs.
    # Canonical source: /v1internal:fetchAvailableModels (PROD endpoint).
    model_mappings = {
        # Claude -- bare runtime IDs (proven 200 with stripped IDs).
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-6": "claude-opus-4-6-thinking",
        "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
        # OpenAI -- catalog ID is bare.
        "gpt-oss-120b-medium": "gpt-oss-120b-medium",
        # Gemini 3.8 Flash -- bare base absent from catalog; default high tier.
        "gemini-3.8-flash": "gemini-3.8-flash-high",
        "gemini-3.8-flash-high": "gemini-3.8-flash-high",
        "gemini-3.8-flash-low": "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium": "gemini-3.8-flash-medium",
        "gemini-3.8-flash-tiered": "gemini-3.8-flash-tiered",
        # Gemini 3.1 Pro -- catalog deprecatedModelIds maps high to successor.
        "gemini-3.1-pro": "gemini-pro-agent",
        "gemini-3.1-pro-preview": "gemini-pro-agent",
        "gemini-3.1-pro-high": "gemini-pro-agent",
        "gemini-3.1-pro-low": "gemini-3.1-pro-low",
        "gemini-pro-agent": "gemini-pro-agent",
        # Gemini 3 Pro (high/low tiers)
        "gemini-3-pro": "gemini-3-pro-high",
        "gemini-3-pro-preview": "gemini-3-pro-high",
        "gemini-3-pro-high": "gemini-3-pro-high",
        "gemini-3-pro-low": "gemini-3-pro-low",
        # Gemini 3 Flash -- note: no bare "gemini-3.1-flash" exists on the API;
        # only "gemini-3.1-flash-lite" and "gemini-3.1-flash-image" do.
        "gemini-3-flash": "gemini-3-flash",
        "gemini-3-flash-agent": "gemini-3-flash-agent",
        "gemini-3.1-flash": "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
        "gemini-3.1-flash-image": "gemini-3.1-flash-image",
        # Gemini 2.5
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.5-flash-thinking": "gemini-2.5-flash-thinking",
        "gemini-2.5-pro": "gemini-2.5-pro",
    }

    return model_mappings.get(model_lower, model_lower)


def _extract_antigravity_text(data: dict, model: str = "") -> str:
    """Extract text from Antigravity response."""
    import proxy
    text_parts = []
    response = data.get("response", data)
    candidates = response.get("candidates", [])
    _mcp = getattr(proxy, "MCP_MODE", False)
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        # Track if we found any regular text; if not, include thought as fallback.
        has_visible_text = False
        for part in parts:
            # Skip thought blocks in proxy mode (NPC callers don't see reasoning).
            # In MCP mode, include thought text so general-purpose callers get reasoning.
            # Workaround: gemini-3-flash series sometimes puts the entire response in a thought block.
            is_thought = part.get("thought") or part.get("thoughtSignature")
            is_g3_flash = "gemini-3" in model and "flash" in model
            if is_thought and not _mcp and not is_g3_flash:
                continue
            text = part.get("text", "")
            if text:
                has_visible_text = True
                text_parts.append(text)
        # If no regular text was found, include thought blocks as last resort.
        if not has_visible_text:
            for part in parts:
                is_thought = part.get("thought") or part.get("thoughtSignature")
                if is_thought:
                    text = part.get("text", "")
                    if text:
                        text_parts.append(text)
    return "".join(text_parts)


async def call_antigravity_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Call Antigravity API directly with multi-account fallback on 503 errors."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.antigravity_auth.is_ready:
        await proxy.antigravity_auth.refresh_if_needed()
        if not proxy.antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready -- run /config/antigravity-login first")

    ag_model = _get_antigravity_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Antigravity {ag_model} ({len(messages)} msgs)")
    start = time.time()

    # Get all available accounts for fallback
    all_accounts = proxy.antigravity_auth._get_active_accounts()
    if not all_accounts:
        raise HTTPException(status_code=503, detail="No Antigravity accounts available")

    last_error = None
    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready:
                ensure_account_session(account)

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": ag_model,
            "request": {
                **request_contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                }
            },
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()
        session = account.session or proxy.create_session()
        owns_session = not account.session

        try:
            async with session.post(
                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                json=payload,
                headers=headers,
            ) as resp:
                elapsed = time.time() - start

                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity {resp.status} (account: {account.email}): {error_body[:300]}")

                    # Handle specific error codes - fallback for transient/recoverable errors
                    if resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES:  # Service unavailable, rate limited, gateway errors, or permission denied
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        # Token expired, try refresh and retry once
                        if await account.refresh_if_needed():
                            headers = account.get_auth_headers()
                            async with session.post(
                                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                                json=payload,
                                headers=headers,
                            ) as retry_resp:
                                if retry_resp.status == 200:
                                    data = await retry_resp.json()
                                    proxy.antigravity_auth.mark_account_success(account)
                                    return _extract_antigravity_text(data, ag_model)
                                elif retry_resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES:
                                    proxy.antigravity_auth.mark_account_error(account, retry_resp.status)
                                    last_error = HTTPException(status_code=retry_resp.status, detail=error_body[:200])
                                    continue  # Try next account
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        continue  # Try next account

                    # For other errors, raise immediately
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])

                data = await resp.json()
                response_text = _extract_antigravity_text(data, ag_model)
                proxy.antigravity_auth.mark_account_success(account)
                logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s, Antigravity, account: {account.email})")
                return response_text
        except HTTPException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity request error for {account.email}: {e}")
            proxy.antigravity_auth.mark_account_error(account, 500)
            last_error = HTTPException(status_code=500, detail=str(e))
        finally:
            if owns_session:
                await session.close()

    # All accounts failed
    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="All Antigravity accounts failed")


async def call_antigravity_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Call Antigravity API with streaming and multi-account fallback on initial errors."""
    import proxy
    _mcp = getattr(proxy, "MCP_MODE", False)
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.antigravity_auth.is_ready:
        await proxy.antigravity_auth.refresh_if_needed()
        if not proxy.antigravity_auth.is_ready:
            error, done = yield_sse_error(
                model, "[Antigravity Error: auth not ready]", status_code=503
            )
            yield error
            yield done
            return

    ag_model = _get_antigravity_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Antigravity {ag_model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }

    # Get all available accounts for fallback
    all_accounts = proxy.antigravity_auth._get_active_accounts()
    if not all_accounts:
        error, done = yield_sse_error(
            model, "[Antigravity Error: no accounts available]", status_code=503
        )
        yield error
        yield done
        return

    last_error_status = None
    last_error_msg = None

    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready:
                ensure_account_session(account)

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": ag_model,
            "request": {
                **request_contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                }
            },
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()

        session = account.session or proxy.create_session()
        owns_session = not account.session

        async def make_stream_request(headers_to_use):
            """Make the streaming request, return (resp, should_retry) tuple."""
            resp = await session.post(
                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse",
                json=payload,
                headers=headers_to_use,
            )
            if resp.status == 401:
                # Token might be expired, try refresh
                if await account.refresh_if_needed():
                    return resp, True  # Signal retry needed
            return resp, False

        try:
            resp, should_retry = await make_stream_request(headers)

            if should_retry:
                # Close the old response and retry with new headers
                await resp.release()
                headers = account.get_auth_headers()
                resp, should_retry = await make_stream_request(headers)

            async with resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity {resp.status} (account: {account.email}): {error_body[:300]}")

                    # Handle specific error codes - try fallback for transient/recoverable errors
                    if resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES:
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        continue  # Try next account

                    # For other errors, send error and stop
                    error, done = yield_sse_error(
                        model,
                        f"[Antigravity Error {resp.status}]",
                        status_code=resp.status,
                    )
                    yield error
                    yield done
                    return

                yield f"data: {json.dumps(role_chunk)}\n\n"

                # Parse Antigravity SSE -> OpenAI SSE
                decoder = IncrementalSSEDecoder()
                got_finish = False
                async for raw_chunk in resp.content.iter_any():
                    for frame in decoder.feed(raw_chunk):
                        parsed = parse_sse_frame(frame)
                        if not parsed.data:
                            continue
                        try:
                            event = json.loads(parsed.data)
                        except json.JSONDecodeError:
                            continue

                        # Extract text from Antigravity response
                        response = event.get("response", event)
                        candidates = response.get("candidates", [])
                        for candidate in candidates:
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])
                            # Track if we found any regular text; if not, include thought as fallback.
                            has_visible_text = False
                            for part in parts:
                                # Skip thought blocks in proxy mode (MCP mode preserves reasoning)
                                # Workaround: gemini-3-flash series sometimes puts the entire response in a thought block.
                                is_thought = part.get("thought") or part.get("thoughtSignature")
                                is_g3_flash = "gemini-3" in model and "flash" in model
                                if is_thought and not _mcp and not is_g3_flash:
                                    continue
                                text = part.get("text", "")
                                if text:
                                    has_visible_text = True
                                    total_chars += len(text)
                                    oai_chunk = {
                                        "id": cmpl_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(oai_chunk)}\n\n"
                            # If no regular text was found, include thought blocks as last resort.
                            if not has_visible_text:
                                for part in parts:
                                    is_thought = part.get("thought") or part.get("thoughtSignature")
                                    if is_thought:
                                        text = part.get("text", "")
                                        if text:
                                            total_chars += len(text)
                                            oai_chunk = {
                                                "id": cmpl_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model,
                                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                            }
                                            yield f"data: {json.dumps(oai_chunk)}\n\n"

                            if candidate.get("finishReason"):
                                got_finish = True

                # Process any trailing unterminated frame from the decoder.
                for frame in decoder.finalize():
                    parsed = parse_sse_frame(frame)
                    if not parsed.data:
                        continue
                    try:
                        event = json.loads(parsed.data)
                    except json.JSONDecodeError:
                        continue
                    response = event.get("response", event)
                    for candidate in response.get("candidates", []):
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        has_visible_text = False
                        for part in parts:
                            is_thought = part.get("thought") or part.get("thoughtSignature")
                            is_g3_flash = "gemini-3" in model and "flash" in model
                            if is_thought and not _mcp and not is_g3_flash:
                                continue
                            text = part.get("text", "")
                            if text:
                                has_visible_text = True
                                total_chars += len(text)
                                oai_chunk = {
                                    "id": cmpl_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(oai_chunk)}\n\n"
                        if not has_visible_text:
                            for part in parts:
                                is_thought = part.get("thought") or part.get("thoughtSignature")
                                if is_thought:
                                    text = part.get("text", "")
                                    if text:
                                        total_chars += len(text)
                                        oai_chunk = {
                                            "id": cmpl_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                        }
                                        yield f"data: {json.dumps(oai_chunk)}\n\n"

                # Single stop chunk + DONE
                stop_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                proxy.antigravity_auth.mark_account_success(account)
                elapsed = time.time() - start
                logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, Antigravity stream, account: {account.email})")
                return  # Success, exit the generator
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity stream error for {account.email}: {e}")
            proxy.antigravity_auth.mark_account_error(account, 500)
            last_error_status = 500
            last_error_msg = str(e)
        finally:
            if owns_session:
                await session.close()

    # All accounts failed
    error_msg = f"[Antigravity Error: All accounts failed - {last_error_status}]"
    if last_error_msg:
        error_msg = f"[Antigravity Error {last_error_status}: {last_error_msg[:100]}]"
    error, done = yield_sse_error(
        model,
        error_msg,
        status_code=last_error_status if last_error_status else 503,
    )
    yield error
    yield done
