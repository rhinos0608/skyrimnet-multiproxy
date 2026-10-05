"""Gemini-format adapter for Antigravity structured tool_use support.

Translates between OpenAI chat completion format (which the existing
Anthropic↔OAI pipeline already speaks) and the Gemini/Antigravity wire
protocol, enabling full tool_use passthrough for Claude Code.

Request flow:
  Anthropic /v1/messages
    → _anthropic_to_oai_structured()  [existing]
    → _oai_payload_to_gemini()         [this module]
    → Antigravity API
    → _gemini_response_to_oai()       [this module]
    → _openai_completion_to_anthropic_message()  [existing]

Streaming:
  Same as above, but _stream_gemini_as_oai_chunks() yields OAI chunk dicts
  that _anthropic_stream_from_openai() translates to Anthropic SSE events.
"""

import json
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.providers.antigravity import ensure_account_session
from proxy_internal.sse_parser import IncrementalSSEDecoder, iter_sse_frames, parse_sse_frame

logger = logging.getLogger("proxy")


_SCHEMA_UNSUPPORTED_FIELDS = {
    "$schema",
    "$id",
    "$defs",
    "$comment",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "propertyNames",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "dependentSchemas",
    "dependentRequired",
    "prefixItems",
    "contains",
    "minContains",
    "maxContains",
    # JSON Schema validation keywords not supported by Gemini API
    "exclusiveMinimum",
    "exclusiveMaximum",
    "const",
}
_ANTIGRAVITY_RETRYABLE_STATUSES = {403, 429, 502, 503, 504}


# ---------------------------------------------------------------------------
# Request-side translation: OAI → Gemini
# ---------------------------------------------------------------------------


def _sanitize_parameters(params):
    """Remove JSON Schema metadata fields not recognized by Gemini API.

    Gemini's functionDeclarations.parameters accepts a subset of OpenAPI /
    JSON Schema but rejects a number of JSON Schema-only keywords such as
    ``$schema`` and ``propertyNames`` anywhere in the tree. Walk the full
    schema recursively so nested combinators like ``anyOf`` and ``items`` are
    sanitized too.
    """
    if isinstance(params, list):
        return [_sanitize_parameters(item) for item in params]
    if not isinstance(params, dict):
        return params

    sanitized = {}
    for key, value in params.items():
        if key in _SCHEMA_UNSUPPORTED_FIELDS:
            continue
        sanitized[key] = _sanitize_parameters(value)
    return sanitized


def _oai_payload_to_gemini(oai_payload: dict) -> dict:
    """Convert an OpenAI chat completion payload to a Gemini request body.

    Returns the *inner* request dict (contents, systemInstruction, tools,
    toolConfig, generationConfig) — the caller wraps it in the Antigravity
    envelope (project, model, request, userAgent, requestId).
    """
    messages = oai_payload.get("messages") or []
    contents: list[dict] = []
    system_parts: list[str] = []

    # Build a tool_call_id → function_name mapping from assistant messages
    # so we can resolve role:"tool" messages that only carry tool_call_id.
    id_to_name: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        # System messages are extracted to systemInstruction.
        if role == "system":
            text = content if isinstance(content, str) else json.dumps(content)
            if text:
                system_parts.append(text)
            continue

        # Assistant messages → model role.
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            parts: list[dict] = []

            # Text content (may be None when tool_calls are present).
            if isinstance(content, str) and content:
                parts.append({"text": content})

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                tc_id = tc.get("id", "")
                # Record mapping for later tool result resolution.
                if tc_id and name:
                    id_to_name[tc_id] = name
                args_raw = fn.get("arguments") or "{}"
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        args = {"_raw_arguments": args_raw}
                else:
                    args = args_raw if isinstance(args_raw, dict) else {}
                fc: dict = {"name": name, "args": args}
                if tc_id:
                    fc["id"] = tc_id
                parts.append({"functionCall": fc})

            if not parts:
                # Empty assistant turn — Gemini requires at least one part.
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
            continue

        # Tool result messages → function role with functionResponse.
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            function_name = id_to_name.get(tool_call_id, "")
            # Content may be a string or JSON string.
            response_content = content if isinstance(content, str) else json.dumps(content)
            # Try to parse as JSON for structured response; fall back to plain text.
            try:
                response_parsed = json.loads(response_content)
            except (json.JSONDecodeError, TypeError):
                response_parsed = {"result": response_content}

            fr: dict = {
                "name": function_name or "unknown_function",
                "response": response_parsed if isinstance(response_parsed, dict) else {"result": response_parsed},
            }
            if tool_call_id:
                fr["id"] = tool_call_id
            part = {"functionResponse": fr}
            contents.append({"role": "function", "parts": [part]})
            continue

        # User messages.
        if role == "user":
            text = content if isinstance(content, str) else json.dumps(content)
            contents.append({"role": "user", "parts": [{"text": text}]})
            continue

        # Fallback: treat as user message with unknown role.
        text = content if isinstance(content, str) else json.dumps(content) if content else ""
        contents.append({"role": "user", "parts": [{"text": text}]})

    request_body: dict = {"contents": contents}

    if system_parts:
        request_body["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }

    # Tools → functionDeclarations.
    oai_tools = oai_payload.get("tools") or []
    if oai_tools:
        declarations = []
        for t in oai_tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            name = fn.get("name", "")
            if not name:
                continue
            decl: dict = {"name": name}
            if fn.get("description"):
                decl["description"] = fn["description"]
            params = fn.get("parameters")
            if params and isinstance(params, dict):
                decl["parameters"] = _sanitize_parameters(params)
            declarations.append(decl)
        if declarations:
            request_body["tools"] = [{"functionDeclarations": declarations}]

    # tool_choice → toolConfig.
    tool_choice = oai_payload.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            mode_map = {"auto": "AUTO", "required": "REQUIRED", "none": "NONE"}
            mode = mode_map.get(tool_choice, "AUTO")
            request_body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and tool_choice.get("function", {}).get("name"):
                request_body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [tool_choice["function"]["name"]],
                    }
                }
            elif tool_choice.get("type") == "any":
                request_body["toolConfig"] = {"functionCallingConfig": {"mode": "REQUIRED"}}
            elif tool_choice.get("type") in ("auto", None):
                request_body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

    # generationConfig.
    gen_config: dict = {}
    max_tokens = oai_payload.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        gen_config["maxOutputTokens"] = max_tokens
    if "temperature" in oai_payload:
        gen_config["temperature"] = oai_payload["temperature"]
    if "top_p" in oai_payload:
        gen_config["topP"] = oai_payload["top_p"]
    stop = oai_payload.get("stop")
    if stop:
        if isinstance(stop, str):
            gen_config["stopSequences"] = [stop]
        elif isinstance(stop, list):
            gen_config["stopSequences"] = stop
    if gen_config:
        request_body["generationConfig"] = gen_config

    return request_body


# ---------------------------------------------------------------------------
# Response-side translation: Gemini → OAI
# ---------------------------------------------------------------------------

_GEMINI_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}


def _gemini_response_to_oai(gemini_response: dict, model: str) -> dict:
    """Convert a Gemini generateContent response dict to an OAI chat completion dict."""
    import proxy
    _mcp = getattr(proxy, "MCP_MODE", False)

    response = gemini_response.get("response", gemini_response)
    candidates = response.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    is_g3_flash = "gemini-3" in model and "flash" in model
    has_visible_text = False
    for part in parts:
        if not isinstance(part, dict):
            continue
        # Skip thought blocks in proxy mode (MCP mode preserves reasoning)
        # Workaround: gemini-3-flash sometimes puts entire response in thought block
        is_thought = part.get("thought") or part.get("thoughtSignature")
        if is_thought and not _mcp:
            continue
        if "text" in part:
            text = part["text"]
            if text:
                has_visible_text = True
                text_parts.append(text)
        if "functionCall" in part:
            fc = part["functionCall"]
            name = fc.get("name", "")
            args = fc.get("args") or {}
            tc_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })

    # If no visible text was found, include thought blocks as last resort.
    # Only for g3-flash (workaround: flash sometimes puts entire response in thought block).
    if not has_visible_text and is_g3_flash:
        for part in parts:
            if not isinstance(part, dict):
                continue
            is_thought = part.get("thought") or part.get("thoughtSignature")
            if is_thought:
                text = part.get("text", "")
                if text:
                    text_parts.append(text)

    message_content = "".join(text_parts) if text_parts else ""
    if not tool_calls and not text_parts:
        message_content = ""

    message: dict = {"role": "assistant", "content": message_content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        # If there are tool calls and no text, set content to None per OAI convention.
        if not message_content:
            message["content"] = None

    finish_reason_raw = candidate.get("finishReason") or "STOP"
    finish_reason = _GEMINI_FINISH_MAP.get(finish_reason_raw, "stop")
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    usage = response.get("usageMetadata") or {}
    prompt_tokens = usage.get("promptTokenCount", 0)
    completion_tokens = usage.get("candidatesTokenCount", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming translation: Gemini SSE → OAI chunk dicts
# ---------------------------------------------------------------------------

async def _stream_gemini_as_oai_chunks(
    resp,
    model: str,
    request_id: str,
    cmpl_id: str,
    created: int,
):
    """Parse Antigravity/Gemini SSE and yield OAI chat.completion.chunk dicts.

    ``resp`` is an aiohttp ClientResponse whose body is an SSE stream of
    ``data: {...}`` events.  This generator yields parsed OAI chunk dicts
    (not SSE strings) so that ``_anthropic_stream_from_openai`` can consume
    them directly.
    """
    import proxy as proxy_mod
    _mcp = getattr(proxy_mod, "MCP_MODE", False)
    total_chars = 0
    finish_reason: Optional[str] = None
    tool_call_index = 0

    # Yield initial role chunk.
    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield role_chunk

    async for frame in iter_sse_frames(resp.content.iter_any()):
        parsed = parse_sse_frame(frame)
        if not parsed.data:
            continue
        try:
            event = json.loads(parsed.data)
        except json.JSONDecodeError:
            continue

        response = event.get("response", event)
        candidates = response.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            is_g3_flash = "gemini-3" in model and "flash" in model
            has_visible_text = False
            for part in parts:
                if not isinstance(part, dict):
                    continue
                is_thought = part.get("thought") or part.get("thoughtSignature")
                if is_thought and not _mcp:
                    continue
                if "text" in part:
                    text = part["text"]
                    if text:
                        has_visible_text = True
                        total_chars += len(text)
                        yield {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                if "functionCall" in part:
                    fc = part["functionCall"]
                    name = fc.get("name", "")
                    args = fc.get("args") or {}
                    tc_id = f"call_{uuid.uuid4().hex[:8]}"
                    yield {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": tool_call_index,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    tool_call_index += 1
            # If no visible text, include thought blocks as last resort.
            # Only for g3-flash (workaround: flash sometimes puts entire response in thought block).
            if not has_visible_text and is_g3_flash:
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    is_thought = part.get("thought") or part.get("thoughtSignature")
                    if is_thought:
                        text = part.get("text", "")
                        if text:
                            total_chars += len(text)
                            yield {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                            }

                fr = candidate.get("finishReason")
                if fr and not finish_reason:
                    finish_reason = _GEMINI_FINISH_MAP.get(fr, "stop")

    # Force tool_calls finish reason if we emitted any function calls.
    if tool_call_index > 0 and (not finish_reason or finish_reason == "stop"):
        finish_reason = "tool_calls"

    # Final stop chunk.
    yield {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
    }


# ---------------------------------------------------------------------------
# Structured call / stream functions
# ---------------------------------------------------------------------------

async def call_antigravity_structured(
    resolved: dict,
    oai_payload: dict,
    request_id: str,
) -> dict:
    """Non-streaming structured call to Antigravity.

    Converts the OAI payload to Gemini format, calls Antigravity, and returns
    an OAI chat completion dict.  Iterates accounts with fallback on
    transient errors (503, 429, 403, 401).
    """
    import proxy

    if not proxy.antigravity_auth.is_ready:
        await proxy.antigravity_auth.refresh_if_needed()
        if not proxy.antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready")

    api_model = resolved["api_model"]
    gemini_request = _oai_payload_to_gemini(oai_payload)

    start = time.time()
    all_accounts = proxy.antigravity_auth._get_active_accounts()
    if not all_accounts:
        raise HTTPException(status_code=503, detail="No Antigravity accounts available")

    last_error: Optional[HTTPException] = None

    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready:
                ensure_account_session(account)

        if not account.is_ready:
            continue

        payload = {
            "project": account.project_id or resolved.get("project_id") or proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": api_model,
            "request": gemini_request,
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()
        session = account.session or proxy.create_session()
        owns_session = not account.session

        try:
            async with session.post(
                resolved["endpoint_url"],
                json=payload,
                headers=headers,
            ) as resp:
                elapsed = time.time() - start

                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity structured {resp.status} (account: {account.email}): {error_body[:500]}")
                    logger.debug(f"[{request_id}] Antigravity structured payload: {json.dumps(payload)[:2000]}")

                    if resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES:
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        continue

                    if resp.status == 401:
                        if await account.refresh_if_needed():
                            headers = account.get_auth_headers()
                            async with session.post(
                                resolved["endpoint_url"],
                                json=payload,
                                headers=headers,
                            ) as retry_resp:
                                if retry_resp.status == 200:
                                    data = await retry_resp.json()
                                    result = _gemini_response_to_oai(data, resolved.get("provider_model", api_model))
                                    proxy.antigravity_auth.mark_account_success(account)
                                    logger.info(f"[{request_id}] <- structured {elapsed:.1f}s (Antigravity, account: {account.email})")
                                    return result
                                elif retry_resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES:
                                    proxy.antigravity_auth.mark_account_error(account, retry_resp.status)
                                    last_error = HTTPException(status_code=retry_resp.status, detail=error_body[:200])
                                    continue
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        continue

                    raise HTTPException(status_code=resp.status, detail=error_body[:500])

                data = await resp.json()
                result = _gemini_response_to_oai(data, resolved.get("provider_model", api_model))
                proxy.antigravity_auth.mark_account_success(account)
                logger.info(f"[{request_id}] <- structured {elapsed:.1f}s (Antigravity, account: {account.email})")
                return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity structured error for {account.email}: {e}")
            proxy.antigravity_auth.mark_account_error(account, 500)
            last_error = HTTPException(status_code=500, detail=str(e))
        finally:
            if owns_session:
                await session.close()

    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="All Antigravity accounts failed")


async def stream_antigravity_structured(
    resolved: dict,
    oai_payload: dict,
    request_id: str,
):
    """Streaming structured call to Antigravity.

    Yields OAI chat.completion.chunk dicts.  Iterates accounts with fallback
    on transient errors.
    """
    import proxy as proxy_mod
    _mcp = getattr(proxy_mod, "MCP_MODE", False)

    if not proxy_mod.antigravity_auth.is_ready:
        await proxy_mod.antigravity_auth.refresh_if_needed()
        if not proxy_mod.antigravity_auth.is_ready:
            err_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resolved.get("provider_model", resolved["api_model"]),
                "choices": [{"index": 0, "delta": {"content": "[Antigravity Error: auth not ready]"}, "finish_reason": None}],
            }
            yield err_chunk
            yield {"id": err_chunk["id"], "object": "chat.completion.chunk",
                   "created": err_chunk["created"], "model": err_chunk["model"],
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            return

    api_model = resolved["api_model"]
    gemini_request = _oai_payload_to_gemini(oai_payload)
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    start = time.time()

    all_accounts = proxy_mod.antigravity_auth._get_active_accounts()
    if not all_accounts:
        err_chunk = {
            "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
            "model": resolved.get("provider_model", api_model),
            "choices": [{"index": 0, "delta": {"content": "[Antigravity Error: No accounts available]"}, "finish_reason": None}],
        }
        yield err_chunk
        yield {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
               "model": err_chunk["model"],
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
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

        payload = {
            "project": account.project_id or resolved.get("project_id") or proxy_mod.ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": api_model,
            "request": gemini_request,
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()

        session = account.session or proxy_mod.create_session()
        owns_session = not account.session

        try:
            resp = await session.post(
                resolved["stream_url"],
                json=payload,
                headers=headers,
            )

            if resp.status == 401:
                await resp.release()
                if await account.refresh_if_needed():
                    headers_retry = account.get_auth_headers()
                    resp = await session.post(
                        resolved["stream_url"],
                        json=payload,
                        headers=headers_retry,
                    )

            async with resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity structured stream {resp.status} (account: {account.email}): {error_body[:300]}")

                    if resp.status in _ANTIGRAVITY_RETRYABLE_STATUSES or resp.status == 401:
                        proxy_mod.antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        continue

                    # Non-retryable error.
                    err_chunk = {
                        "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                        "model": resolved.get("provider_model", api_model),
                        "choices": [{"index": 0, "delta": {"content": f"[Antigravity Error {resp.status}]"}, "finish_reason": None}],
                    }
                    yield err_chunk
                    yield {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                           "model": err_chunk["model"],
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    return

                logger.info(f"[{request_id}] -> Antigravity structured stream {api_model}")
                async for chunk in _stream_gemini_as_oai_chunks(
                    resp, resolved.get("provider_model", api_model), request_id, cmpl_id, created
                ):
                    yield chunk

                proxy_mod.antigravity_auth.mark_account_success(account)
                elapsed = time.time() - start
                logger.info(f"[{request_id}] <- structured stream done {elapsed:.1f}s (Antigravity, account: {account.email})")
                return
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity structured stream error for {account.email}: {e}")
            proxy_mod.antigravity_auth.mark_account_error(account, 500)
            last_error_status = 500
            last_error_msg = str(e)
        finally:
            if owns_session:
                await session.close()

    # All accounts failed.
    error_msg = f"[Antigravity Error: All accounts failed - {last_error_status}]"
    if last_error_msg:
        error_msg = f"[Antigravity Error {last_error_status}: {last_error_msg[:100]}]"
    err_chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
        "model": resolved.get("provider_model", api_model),
        "choices": [{"index": 0, "delta": {"content": error_msg}, "finish_reason": None}],
    }
    yield err_chunk
    yield {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
           "model": err_chunk["model"],
           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
