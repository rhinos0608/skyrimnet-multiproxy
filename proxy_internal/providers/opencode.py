"""OpenCode (Zen + Go) OpenAI-compatible provider."""

import asyncio
import json
import logging
import os
import re
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

_reasoning_strip_warned = False
_OPENCODE_SESSION_ID = str(uuid.uuid4())
_OPENCODE_PROJECT_ID = str(uuid.uuid4())
_OPENCODE_USER_AGENT = "opencode/latest/1.3.15/cli"
_OPENCODE_GO_USER_AGENT = "skyrimnet-codex-proxy/1.0"

# Go backend routes models to model-dependent endpoints (/chat/completions,
# /messages, /responses) per https://opencode.ai/docs/go/ model table.
# Chat-compatible: GLM glm-5.3 / glm-5.3-flash / glm-5.2, Kimi k3 / k2.7-code /
# k2.6. Known non-chat (must fail loudly, never misroute to chat):
# /messages-only: MiniMax M3 / M2.7, Qwen3.8 Max/Flash, Qwen3.7 Plus;
# /responses-only: Muse Spark 1.3 / 1.2 Contributor, GPT-6 Luna, GPT-5.6 Luna,
# Grok 4.7 / 4.6.
# Matchers stay version-pinned (e.g. "grok-4", "gpt-5.6", "qwen3.7") so
# unrelated chat families never false-positive.
_GO_NON_CHAT_FAMILIES = (
    "minimax", "m2.7", "-m3", "m3-", "/m3",
    "qwen3.8", "qwen-3.8", "qwen3-8",
    "qwen3.7", "qwen-3.7", "qwen3-7",
    "muse-spark", "muse_spark", "muse spark",
    "gpt-6", "gpt6",
    "gpt-5.6", "gpt-5_6", "gpt5.6",
    "grok-4", "grok_4", "grok 4",
)


def _require_go_chat_model(api_model: str) -> None:
    lowered = api_model.lower()
    for family in _GO_NON_CHAT_FAMILIES:
        if family in lowered:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"OpenCode Go model '{api_model}' is not chat-compatible "
                    "(requires /messages or /responses endpoint); "
                    "use a documented /chat/completions model (e.g. glm-5.3, kimi-k3, mimo-v2.6-flash)"
                ),
            )


def _build_opencode_headers(api_key: str, plan: str = "Zen", session_id: Optional[str] = None) -> dict[str, str]:
    """Build request headers. Zen sends CLI identity headers; Go uses an
    honest proxy User-Agent plus a stable per-conversation x-opencode-session
    (fresh UUID fallback per request when caller supplies none).
    """
    if plan == "Go":
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": _OPENCODE_GO_USER_AGENT,
            "x-opencode-session": session_id or str(uuid.uuid4()),
        }
    base = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _OPENCODE_USER_AGENT,
    }
    if plan == "Zen":
        base.update({
            "x-opencode-client": "cli",
            "x-opencode-session": _OPENCODE_SESSION_ID,
            "x-opencode-project": _OPENCODE_PROJECT_ID,
            "x-opencode-request": str(uuid.uuid4()),
        })
    return base


def _cli_file_key(entry: str) -> Optional[str]:
    """Return the raw key for ``entry`` in the OpenCode CLI auth file.

    Read-only metadata use: lets the resolver tell a CLI-file auto-loaded
    global apart from explicit operator config. Never logged; None on any
    problem.
    """
    try:
        path = os.path.join(
            os.path.expanduser("~"), ".local", "share", "opencode", "auth.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        record = data.get(entry)
        if not isinstance(record, dict) or record.get("type") != "api":
            return None
        key = record.get("key")
        return key.strip() if isinstance(key, str) and key.strip() else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _prefer_discovery(global_key: Optional[str], discovered: Optional[str]) -> bool:
    """True when the plan-matched discovered key should beat the global.

    Only when they disagree AND the global is exactly what the CLI auth
    file holds (i.e. startup auto-load, not explicit operator config).
    An explicitly configured global always wins.
    """
    if not discovered or not global_key or discovered == global_key:
        return False
    return global_key in (_cli_file_key("opencode-go"), _cli_file_key("opencode"))


def _resolve_opencode(model: str):
    """Resolve OpenCode model prefix to (api_model, base_url, api_key, plan_name).

    Key priority: OPENCODE_API_KEY env > explicit proxy config for that plan
    > plan-specific proxy global > plan-matched discovery, EXCEPT that a
    global which merely mirrors the CLI auth file (startup auto-load, incl.
    the Zen<->Go cross-fill) yields to a disagreeing plan-matched discovered
    key — the plans can hold distinct keys and the other plan's key yields
    upstream 403. Resolved dynamically per call so late logins are picked
    up without restart.

    Guards against double-prefixed models like 'opencode:xiaomi:mimo-v2-pro' —
    if the stripped model still contains another provider prefix, that's a config
    error on the SkyrimNet side.
    """
    import proxy
    env_key = os.environ.get("OPENCODE_API_KEY") or None
    cfg = getattr(proxy, "_cfg", None) or {}
    from proxy_internal.credential_discovery import discover_opencode_key
    if model.lower().startswith("opencode-go:"):
        key = (env_key or cfg.get("opencode_go_api_key")
               or proxy.opencode_go_api_key or proxy.opencode_api_key)
        discovered = discover_opencode_key("Go")
        if _prefer_discovery(key, discovered):
            key = discovered
        elif not key:
            key = discovered
        return model[len("opencode-go:"):], proxy.OPENCODE_GO_URL, key, "Go"
    key = (env_key or cfg.get("opencode_api_key")
           or proxy.opencode_api_key or proxy.opencode_go_api_key)
    discovered = discover_opencode_key("Zen")
    if _prefer_discovery(key, discovered):
        key = discovered
    elif not key:
        key = discovered
    return model[len("opencode:"):], proxy.OPENCODE_ZEN_URL, key, "Zen"


_OPENCODE_UPSTREAM_REASON_LEN = 300

_OPENCODE_SECRET_SHAPED_RE = re.compile(
    r"sk-[A-Za-z0-9-_]{8,}|gho_[A-Za-z0-9_]+|ghu_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"(?:api[_-]?key|bearer|token)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _scrub_opencode_reason(body_text: str, limit: int = _OPENCODE_UPSTREAM_REASON_LEN) -> str:
    """Bounded single-line excerpt of an upstream error body for debug logs.

    Response body only — never request headers or credentials. Newlines
    collapsed, capped at ``limit`` chars, and key-shaped values replaced
    so secrets stay out of logs. Debug-level only: existing redaction
    keeps raw upstream text out of error details and error logs.
    """
    flat = " ".join((body_text or "").split())
    flat = _OPENCODE_SECRET_SHAPED_RE.sub("[redacted]", flat)
    return flat[:limit]


async def call_opencode_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenCode API (OpenAI-compatible), collect full response."""
    import proxy
    api_model, base_url, api_key, plan = _resolve_opencode(model)
    if not api_key:
        raise HTTPException(status_code=503, detail=f"OpenCode {plan} API key not configured — add it via /config/opencode-key")
    endpoint = f"{base_url}/chat/completions"
    go_session = extra_params.pop("opencode_session_id", None)
    if plan == "Go":
        _require_go_chat_model(api_model)
    headers = _build_opencode_headers(api_key, plan, session_id=go_session)

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages)}
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for OpenCode "
            "(upstream does not support it); 'reasoning_effort' stripped for Go "
            "(Moonshot AI backend requires it absent with thinking disabled)"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})
    # Moonshot AI (OpenCode Go backend) requires reasoning_effort absent when
    # thinking is disabled — otherwise it rejects tool-call messages.
    if plan == "Go":
        payload.pop("reasoning_effort", None)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenCode ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                if plan == "Go":
                    logger.debug(f"[{request_id}] OpenCode Go error reason: {_scrub_opencode_reason(error_body)}")
                    logger.error(f"[{request_id}] OpenCode Go error status={resp.status}")
                    if resp.status in (401, 403):
                        raise HTTPException(status_code=resp.status, detail=f"OpenCode Go auth failed (status {resp.status})")
                    if resp.status == 429:
                        raise HTTPException(status_code=429, detail="OpenCode Go rate limit exceeded")
                    raise HTTPException(status_code=resp.status, detail=f"OpenCode Go request failed (status {resp.status})")
                # Mirror the Go path above: the upstream body can echo key-shaped
                # material, so it is scrubbed into a DEBUG line and never forwarded
                # to the caller as part of the HTTP error detail.
                logger.debug(f"[{request_id}] OpenCode {plan} error reason: {_scrub_opencode_reason(error_body)}")
                logger.error(f"[{request_id}] OpenCode {plan} error status={resp.status}")
                if resp.status in (401, 403):
                    raise HTTPException(status_code=401, detail=f"OpenCode {plan} auth failed (status {resp.status})")
                if resp.status == 429:
                    raise HTTPException(status_code=429, detail="OpenCode rate limit exceeded")
                raise HTTPException(status_code=resp.status, detail=f"OpenCode {plan} request failed (status {resp.status})")
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="OpenCode returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] OpenCode reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_completion_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                return content
                raise HTTPException(status_code=500, detail="OpenCode returned empty content")
            return content

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"OpenCode {api_model}", request_id=request_id)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="OpenCode request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, OpenCode)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_opencode_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenCode API with streaming, passthrough SSE directly."""
    import proxy
    api_model, base_url, api_key, plan = _resolve_opencode(model)
    if not api_key:
        err, done = yield_sse_error(model, f"[OpenCode {plan} Error: API key not configured]")
        yield err; yield done
        return

    endpoint = f"{base_url}/chat/completions"
    go_session = extra_params.pop("opencode_session_id", None)
    if plan == "Go":
        _require_go_chat_model(api_model)
    headers = _build_opencode_headers(api_key, plan, session_id=go_session)

    # Use max_completion_tokens for streaming (matches reference SDK behaviour)
    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    # OpenCode proxies to various backends — strip thinking/reasoning to avoid errors.
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for OpenCode "
            "(upstream does not support it); 'reasoning_effort' stripped for Go "
            "(Moonshot AI backend requires it absent with thinking disabled)"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})
    # Moonshot AI (OpenCode Go backend) requires reasoning_effort absent when
    # thinking is disabled — otherwise it rejects tool-call messages.
    if plan == "Go":
        payload.pop("reasoning_effort", None)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenCode ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session, endpoint,
                json_payload=payload, headers=headers,
                operation="OpenCode", request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[OpenCode Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "OpenCode", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] OpenCode streaming error: {e}")
            err, done = yield_sse_error(model, f"[OpenCode Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
