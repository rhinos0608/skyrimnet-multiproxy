"""Command Code CLI provider — calls `cmd --print` subprocess."""

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from typing import Optional

import aiohttp
from aiohttp import web
from fastapi import HTTPException

from proxy_internal.interceptor_utils import SessionBindingMixin, content_matches_text
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

_CMD_API_BASE = "https://api.commandcode.ai"
_CMD_UPSTREAM_SESSION = web.AppKey("cmd_upstream_session", aiohttp.ClientSession)
_CMD_UPSTREAM_TIMEOUT = web.AppKey("cmd_upstream_timeout", aiohttp.ClientTimeout)

# Fields to strip from cmd's /alpha/generate request body.
# These are CLI-originated context (skills, tool definitions) that must
# not reach the upstream model.  Required fields like config/memory are
# kept with safe defaults.
_CMD_STRIP_FIELDS = {"skills", "taste", "system_prompt_breakdown"}
_CMD_PARAMS_STRIP_FIELDS = {"tools"}


def _is_parent_generation(payload: dict, expected_prompt: str) -> bool:
    params = payload.get("params")
    messages = params.get("messages") if isinstance(params, dict) else None
    if not isinstance(messages, list):
        messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and content_matches_text(message.get("content"), expected_prompt)
        for message in messages
    )


class _CmdInterceptorScope(SessionBindingMixin):
    def __init__(self, expected_prompt: str):
        self.expected_prompt = expected_prompt
        self.session_id: Optional[str] = None

    def payload_matches(self, payload: dict) -> bool:
        return _is_parent_generation(payload, self.expected_prompt)


_CMD_INTERCEPTOR_SCOPE = web.AppKey("cmd_interceptor_scope", _CmdInterceptorScope)


async def _cmd_interceptor_handler(request):
    """MITM handler for cmd CLI requests.

    Strips CLI-originated system prompts, tool definitions, skills, and
    project context from the request body before forwarding to the real API.
    """
    body = await request.read()
    # Drop hop-by-hop and body-framing headers — aiohttp recomputes them
    # after we mutate the body.  Forwarding stale Content-Length after
    # stripping CLI context would trigger an ECONNRESET upstream.
    hop_by_hop = {"host", "content-length", "transfer-encoding", "connection",
                  "keep-alive", "proxy-authenticate", "proxy-authorization",
                  "te", "trailer", "upgrade", "accept-encoding"}
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in hop_by_hop}

    real_url = f"{_CMD_API_BASE}{request.path_qs}"

    # Child tools inherit COMMANDCODE_API_URL. Only strip the parent invocation;
    # unrelated or unrecognized generation payloads pass through unchanged.
    # Only the CLI's parent generation endpoint can carry parent-only context.
    # Provider relay requests must never bind this invocation's session.
    if request.path == "/alpha/generate":
        try:
            parsed = json.loads(body) if body else {}
            scope = request.app[_CMD_INTERCEPTOR_SCOPE]
            session_id = request.headers.get("x-session-id")
            if (
                isinstance(parsed, dict)
                and scope.payload_matches(parsed)
                and scope.bind_session(session_id)
            ):
                for field in _CMD_STRIP_FIELDS:
                    parsed.pop(field, None)

                params = parsed.get("params", {})
                if isinstance(params, dict):
                    for field in _CMD_PARAMS_STRIP_FIELDS:
                        params.pop(field, None)
                    params.pop("system", None)
                    if params:
                        parsed["params"] = params
                    else:
                        parsed.pop("params", None)

                body = json.dumps(parsed).encode("utf-8")
        except (json.JSONDecodeError, TypeError):
            pass

    # Forward through the session shared by this CLI invocation so sequential
    # control-plane and generation requests can reuse DNS/TCP/TLS state.
    # The per-invocation 30s-capped timeout is applied per upstream request
    # (the borrowed shared session carries aiohttp's default timeout).
    session = request.app[_CMD_UPSTREAM_SESSION]
    timeout = request.app[_CMD_UPSTREAM_TIMEOUT]
    try:
        async with session.request(
            method=request.method,
            url=real_url,
            data=body,
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"cmd interceptor upstream error: {type(e).__name__}")
        return web.Response(
            status=502, content_type="application/json",
            body=b'{"error":"proxy upstream error"}',
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if "closed" in msg or "session" in msg:
            logger.warning("cmd interceptor session-lifecycle error")
            return web.Response(
                status=502, content_type="application/json",
                body=b'{"error":"proxy upstream error"}',
            )
        raise


async def _run_with_interceptor(
    cmd_args: list[str], env: dict, *, expected_prompt: str, prompt_input: Optional[bytes] = None, timeout: int = 180
) -> tuple[bytes, bytes]:
    """Run cmd through a local MITM proxy scoped to one parent prompt.

    Reuses ``proxy.third_party_session`` when one is open (production
    lifespan creates it) so cmd shares the proxy's connection pool; the
    upstream timeout is applied per request. When no shared session exists
    (isolated tests, non-lifespan use) an owned fallback session is created
    and closed by this invocation. The web.Application and
    ``_CmdInterceptorScope`` stay per-invocation, so cookies and session
    bindings never leak across CLI runs.
    """
    import proxy
    upstream_total = float(timeout)
    client_timeout = aiohttp.ClientTimeout(
        total=upstream_total,
        sock_connect=min(10.0, upstream_total),
        sock_read=upstream_total,
    )
    shared_session = getattr(proxy, "third_party_session", None)
    owns_session = False
    if shared_session is None or shared_session.closed:
        shared_session = getattr(proxy.auth, "session", None)
    if shared_session is None or shared_session.closed:
        shared_session = aiohttp.ClientSession(timeout=client_timeout)
        owns_session = True
    try:
        app = web.Application()
        app[_CMD_UPSTREAM_SESSION] = shared_session
        app[_CMD_UPSTREAM_TIMEOUT] = client_timeout
        app[_CMD_INTERCEPTOR_SCOPE] = _CmdInterceptorScope(expected_prompt)
        app.router.add_route("*", "/{path_info:.*}", _cmd_interceptor_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]

        proc = None
        try:
            proxy_env = dict(env)
            proxy_env["COMMANDCODE_API_URL"] = f"http://127.0.0.1:{port}"
            proxy_env["COMMANDCODE_SANDBOX"] = "true"
            proxy_env["DO_NOT_TRACK"] = "1"

            create_kwargs = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": proxy_env,
            }
            if prompt_input is not None:
                create_kwargs["stdin"] = asyncio.subprocess.PIPE
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                **create_kwargs,
            )
            if prompt_input is None:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            else:
                stdout, stderr = await asyncio.wait_for(proc.communicate(input=prompt_input), timeout=timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            try:
                if proc is not None and proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except ProcessLookupError:
                        pass
                    except asyncio.TimeoutError:
                        logger.warning("Command Code subprocess reap timed out")
            finally:
                await runner.cleanup()

        return stdout, stderr
    finally:
        if owns_session:
            await shared_session.close()



def _flatten_message_content(content) -> str:
    """Flatten structured content blocks to plain text.

    Handles:
      - str: pass through
      - list[text block]: join text fields
      - list[tool_use block]: convert to [tool_use: name input] placeholder
      - list[tool_result block]: convert to [tool_result: ...] placeholder
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(f"[tool_use: {block.get('name')} {json.dumps(block.get('input', {}))}]")
            elif btype == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(b.get("text", "") for b in result_content if isinstance(b, dict))
                parts.append(f"[tool_result: {result_content}]")
        return "\n".join(parts)
    return str(content) if content else ""


def _build_command_code_prompt(system_prompt: Optional[str], messages: list) -> tuple[Optional[str], list[dict]]:
    """Build a clean text prompt for the Command Code CLI.

    Returns (system_prompt, flat_messages) where tool/tool_use/tool_result blocks
    have been converted to plain-text placeholders so the CLI sees clean input.
    """
    # Build system: only user-supplied system prompt (no tool hints from CLI)
    system_parts: list[str] = []
    if system_prompt:
        system_parts.append(system_prompt)

    new_system = "\n\n".join(filter(None, system_parts)) or None

    # Flatten messages: strip tool blocks, convert to plain text
    flat_messages: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        flat_messages.append({"role": role, "content": _flatten_message_content(content)})

    return new_system, flat_messages


# Allowlisted open-source models (short names — CLI resolves the full path)
_COMMAND_CODE_ALLOWED_MODELS: set[str] = {
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen3.6-max-preview",
    "qwen3.6-plus",
    "qwen3.7-max",
    "kimi-k2.6",
    "kimi-k2.5",
    "glm-5.1",
    "glm-5",
    "minimax-m2.7",
    "minimax-m2.5",
    "step-3.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
}


def _resolve_cmd_model(model: str) -> str:
    """Strip the 'command_code:' prefix to get the upstream model name."""
    return model[len("command_code:"):]


def _get_cmd_path() -> str:
    """Return the path to the cmd binary."""
    path = shutil.which("cmd")
    if not path:
        raise HTTPException(status_code=503, detail="Command Code CLI not installed — install from commandcode.ai")
    return path


async def _is_cmd_authenticated() -> bool:
    """Check if cmd CLI has valid auth via `cmd status --json`."""
    import proxy
    cmd_path = proxy.COMMAND_CODE_PATH
    if not cmd_path:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            cmd_path, "status", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            return False
        data = json.loads(stdout.decode("utf-8", errors="replace"))
        return bool(data.get("authenticated"))
    except Exception:
        return False


async def _fetch_cmd_models() -> list[dict]:
    """Fetch available models via `cmd --list-models` (pass-through).

    Every well-formed `vendor/model` line from the CLI is returned as-is.
    """
    import proxy
    cmd_path = proxy.COMMAND_CODE_PATH
    if not cmd_path:
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            cmd_path, "--list-models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            return []
        result = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            # Skip blank lines, category headers, and description-only lines
            if not line or "Available models" in line or "Docs:" in line or line.startswith("-"):
                continue
            parts = line.split(None, 1)
            full_id = parts[0] if parts else ""
            if not full_id or "/" not in full_id:
                continue
            # Extract short name (after last /); pass through all CLI models.
            short = full_id.rsplit("/", 1)[-1]
            if not short:
                continue
            result.append({
                    "id": f"command_code:{short}",
                    "object": "model",
                    "owned_by": "command_code",
                })
        if not result:
            # Fallback: return allowlist directly
            for mid in sorted(_COMMAND_CODE_ALLOWED_MODELS):
                result.append({
                    "id": f"command_code:{mid}",
                    "object": "model",
                    "owned_by": "command_code",
                })
        return result
    except Exception:
        return [
            {"id": f"command_code:{mid}", "object": "model", "owned_by": "command_code"}
            for mid in sorted(_COMMAND_CODE_ALLOWED_MODELS)
        ]


async def call_command_code_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Call Command Code CLI subprocess through an MITM proxy that strips CLI context."""
    import proxy
    cmd_path = proxy.COMMAND_CODE_PATH
    if not cmd_path:
        raise HTTPException(status_code=503, detail="Command Code CLI not installed — install from commandcode.ai")

    api_model = _resolve_cmd_model(model)

    call_system, flat_messages = _build_command_code_prompt(system_prompt, messages)

    # Build prompt from messages
    prompt_parts = []
    if call_system:
        prompt_parts.append(f"System: {call_system}")
    for m in flat_messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Command Code CLI ({api_model}, {len(flat_messages)} msgs, via MITM proxy)")
    start = time.time()

    try:
        cmd_args = [cmd_path, "-m", api_model, "--print",
                    "--trust", "--skip-onboarding", "--no-auto-update"]
        env = os.environ.copy()
        stdout, stderr = await _run_with_interceptor(
            cmd_args, env, expected_prompt=full_prompt,
            prompt_input=full_prompt.encode("utf-8")
        )

        elapsed = time.time() - start

        raw_output = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if not raw_output:
            logger.warning(f"[{request_id}] Command Code CLI returned empty output")
            raise HTTPException(status_code=500, detail="Command Code CLI returned empty response")

        logger.info(f"[{request_id}] <- {len(raw_output)} chars ({elapsed:.1f}s)")
        return raw_output

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Command Code CLI error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Command Code CLI error")


async def call_command_code_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Call Command Code CLI subprocess through an MITM proxy and yield response as SSE chunks.

    The MITM proxy strips CLI-originated skills, tool definitions, and system
    prompts from the outgoing API request before forwarding to the upstream.
    """
    import proxy
    cmd_path = proxy.COMMAND_CODE_PATH
    if not cmd_path:
        err, done = yield_sse_error(model, "[Command Code Error: CLI not installed]")
        yield err
        yield done
        return

    api_model = _resolve_cmd_model(model)

    call_system, flat_messages = _build_command_code_prompt(system_prompt, messages)

    # Build prompt from messages
    prompt_parts = []
    if call_system:
        prompt_parts.append(f"System: {call_system}")
    for m in flat_messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Command Code CLI ({api_model}, {len(flat_messages)} msgs, stream, via MITM proxy)")
    start = time.time()

    # Emit role chunk
    role_chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    try:
        cmd_args = [cmd_path, "-m", api_model, "--print",
                    "--trust", "--skip-onboarding", "--no-auto-update"]
        env = os.environ.copy()
        stdout, stderr = await _run_with_interceptor(
            cmd_args, env, expected_prompt=full_prompt,
            prompt_input=full_prompt.encode("utf-8")
        )

        elapsed = time.time() - start
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if not raw_output:
            logger.warning(f"[{request_id}] Command Code CLI returned empty output")
            err_chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": "[Command Code CLI error: empty response]"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            stop_chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream the response in chunks for a realistic streaming feel
        chunk_size = 20
        for i in range(0, len(raw_output), chunk_size):
            chunk_text = raw_output[i:i+chunk_size]
            oai_chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(oai_chunk)}\n\n"

        logger.info(f"[{request_id}] <- {len(raw_output)} chars ({elapsed:.1f}s, simulated stream)")
    except Exception as e:
        logger.error(f"[{request_id}] Command Code CLI streaming error: {type(e).__name__}")
        err_chunk = {
            "id": cmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": "[Command Code CLI error]"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"

    # Emission stop chunk
    stop_chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


COMMAND_CODE_API_PREFIX = "command_api:"
COMMAND_CODE_API_BASE = "https://api.commandcode.ai/provider/v1"
COMMAND_CODE_API_CHAT_URL = f"{COMMAND_CODE_API_BASE}/chat/completions"
_UPGRADE_REQUIRED_CODE = "upgrade_required"
_UPSTREAM_EXCERPT_LEN = 200


def is_command_code_api_model(model: str) -> bool:
    return isinstance(model, str) and model.startswith(COMMAND_CODE_API_PREFIX)


def _resolve_command_api_model(model: str) -> str:
    raw = model[len(COMMAND_CODE_API_PREFIX):]
    if "/" not in raw or raw.startswith("/") or raw.endswith("/"):
        raise HTTPException(
            status_code=400,
            detail="Command Code API needs a full chat-compatible model id "
            "(vendor/model); short Claude-style ids are not sent to /chat/completions",
        )
    return raw


def _command_api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    import proxy
    resolver = getattr(proxy, "_resolve_command_code_api_key", None)
    if callable(resolver):
        key = resolver()
        if key:
            return key
    key = getattr(proxy, "command_code_api_key", None)
    if key:
        return key
    raise HTTPException(status_code=503, detail="Command Code API key not configured")


def _upstream_excerpt(text: str, limit: int = _UPSTREAM_EXCERPT_LEN) -> str:
    return (text or "")[:limit]


_UPSTREAM_REASON_LEN = 300

_SECRET_SHAPED_RE = re.compile(
    r"sk-[A-Za-z0-9-_]{8,}|gho_[A-Za-z0-9_]+|ghu_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"(?:api[_-]?key|bearer|token)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _scrub_upstream_reason(body_text: str, limit: int = _UPSTREAM_REASON_LEN) -> str:
    """Bounded single-line excerpt of an upstream error body for debug logs.

    Response body only — never request headers or credentials. Newlines
    collapsed, capped at ``limit`` chars, and key-shaped values replaced
    so secrets stay out of logs. Debug-level only: existing redaction
    keeps raw upstream text out of error details and warning+ logs.
    """
    flat = " ".join((body_text or "").split())
    flat = _SECRET_SHAPED_RE.sub("[redacted]", flat)
    return flat[:limit]


def _is_upgrade_required(body_text: str) -> bool:
    try:
        data = json.loads(body_text or "")
    except (ValueError, TypeError):
        return False
    if isinstance(data, dict):
        if data.get("code") == _UPGRADE_REQUIRED_CODE:
            return True
        err = data.get("error")
        if isinstance(err, dict) and err.get("code") == _UPGRADE_REQUIRED_CODE:
            return True
    return False


def _cli_fallback_target(api_model: str) -> Optional[str]:
    """Return full vendor/model for CLI fallback, or None if ineligible.

    Passes through any well-formed vendor/model; the 403 upgrade_required
    handler additionally gates on the allowlisted short name so unknown
    models fail loudly with the GOAT-plan 403 instead of retrying via CLI.
    """
    short = api_model.rsplit("/", 1)[-1].strip()
    if short and "/" in api_model.strip() and not api_model.startswith("/") and not api_model.endswith("/"):
        return api_model
    return None


def _is_cli_supported_model(api_model: str) -> bool:
    """True when the short model name is CLI-allowlisted for 403 fallback."""
    return api_model.rsplit("/", 1)[-1].strip() in _COMMAND_CODE_ALLOWED_MODELS


def _command_api_session():
    import proxy
    auth_session = getattr(getattr(proxy, "auth", None), "session", None)
    session = proxy.third_party_session or auth_session or proxy.create_session()
    owns = session is not proxy.third_party_session and session is not auth_session
    return session, owns


class _CliFallbackSignal(Exception):
    """Internal: API hit upgrade_required and CLI fallback is eligible."""

    def __init__(self, cli_model: str):
        super().__init__(cli_model)
        self.cli_model = cli_model


def _upgrade_error_detail(target: Optional[str], cli_installed: bool) -> str:
    base = ("Command Code API returned 403 upgrade_required: "
            "this model needs the $10 GOAT plan")
    if target is not None:
        if cli_installed:
            return base + " — retried via local CLI"
        return (base + f" for '{target}'; local CLI fallback unavailable "
                "— install the cmd CLI from commandcode.ai or upgrade the plan")
    return base + " — upgrade the plan or use a CLI-supported model"


def _command_api_messages(system_prompt: Optional[str], messages: list) -> list:
    """Build chat-completions messages, preserving caller system content."""
    _, flat = _build_command_code_prompt(system_prompt, messages)
    out: list = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    out.extend(flat)
    return out


async def call_command_code_api_direct(system_prompt: Optional[str], messages: list,
                                       model: str, max_tokens: int,
                                       api_key: Optional[str] = None,
                                       **extra_params) -> str:
    import proxy
    api_model = _resolve_command_api_model(model)
    key = _command_api_key(api_key)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = {"model": api_model,
               "messages": _command_api_messages(system_prompt, messages),
               "max_tokens": max_tokens}
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Command Code API ({api_model}, {len(messages)} msgs)")
    start = time.time()
    session, owns_session = _command_api_session()

    async def _do_call() -> str:
        async with session.post(COMMAND_CODE_API_CHAT_URL, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if not choices:
                    raise HTTPException(status_code=502, detail="Command Code API returned empty response")
                content = _flatten_message_content((choices[0].get("message", {}) or {}).get("content"))
                if not content:
                    raise HTTPException(status_code=502, detail="Command Code API returned empty content")
                return content
            body_text = await resp.text()
            if resp.status == 403 and _is_upgrade_required(body_text):
                target = _cli_fallback_target(api_model)
                if target is not None and _is_cli_supported_model(api_model) and proxy.COMMAND_CODE_PATH:
                    raise _CliFallbackSignal(target)
                if not _is_cli_supported_model(api_model):
                    target = None
                logger.warning(f"[{request_id}] Command Code API 403 upgrade_required")
                raise HTTPException(status_code=403,
                                    detail=_upgrade_error_detail(target, bool(proxy.COMMAND_CODE_PATH)))
            if resp.status == 403:
                logger.debug(f"[{request_id}] Command Code API 403 reason: {_scrub_upstream_reason(body_text)}")
                logger.warning(f"[{request_id}] Command Code API 403")
                raise HTTPException(status_code=403, detail="Command Code API request forbidden")
            logger.debug(f"[{request_id}] Command Code API {resp.status} reason: {_scrub_upstream_reason(body_text)}")
            logger.warning(f"[{request_id}] Command Code API {resp.status}")
            raise HTTPException(status_code=resp.status, detail=f"Command Code API error ({resp.status})")

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"Command Code API {api_model}",
                                              request_id=request_id)
        except _CliFallbackSignal as sig:
            logger.info(f"[{request_id}] Command Code API 403 upgrade_required, CLI fallback")
            return await call_command_code_direct(
                system_prompt, messages, f"command_code:{sig.cli_model}", max_tokens, **extra_params)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Command Code API request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Command Code API)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_command_code_api_streaming(system_prompt: Optional[str], messages: list,
                                          model: str, max_tokens: int,
                                          api_key: Optional[str] = None,
                                          **extra_params):
    import proxy
    api_model = _resolve_command_api_model(model)
    key = _command_api_key(api_key)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = {"model": api_model,
               "messages": _command_api_messages(system_prompt, messages),
               "max_tokens": max_tokens, "stream": True}
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Command Code API stream ({api_model}, {len(messages)} msgs)")
    session, owns_session = _command_api_session()
    resp_cm = None
    try:
        try:
            resp_cm = session.post(COMMAND_CODE_API_CHAT_URL, json=payload, headers=headers)
            resp = await resp_cm.__aenter__()
        except asyncio.TimeoutError:
            err, done = yield_sse_error(model, "[Command Code API timeout]")
            yield err; yield done
            return
        if resp.status == 403:
            body_text = await resp.text()
            await resp_cm.__aexit__(None, None, None)
            resp_cm = None
            if _is_upgrade_required(body_text):
                target = _cli_fallback_target(api_model)
                if target is not None and _is_cli_supported_model(api_model) and proxy.COMMAND_CODE_PATH:
                    logger.info(f"[{request_id}] Command Code API 403 upgrade_required, CLI fallback")
                    async for chunk in call_command_code_streaming(
                            system_prompt, messages, f"command_code:{target}", max_tokens, **extra_params):
                        yield chunk
                    return
                err, done = yield_sse_error(
                    model, f"[{_upgrade_error_detail(target, bool(proxy.COMMAND_CODE_PATH))}]")
                yield err; yield done
                return
            logger.warning(f"[{request_id}] Command Code API 403")
            err, done = yield_sse_error(model, "[Command Code API forbidden]")
            yield err; yield done
            return
        if resp.status != 200:
            try:
                await resp.text()
            except Exception:
                pass
            logger.warning(f"[{request_id}] Command Code API {resp.status}")
            await resp_cm.__aexit__(None, None, None)
            resp_cm = None
            err, done = yield_sse_error(model, f"[Command Code API error ({resp.status})]")
            yield err; yield done
            return
        try:
            async for raw_line in resp.content:
                line = (raw_line.decode("utf-8", errors="replace")
                        if isinstance(raw_line, (bytes, bytearray)) else raw_line)
                for sub in line.splitlines():
                    s = sub.strip()
                    if not s:
                        continue
                    if s.startswith("data: [DONE]"):
                        yield "data: [DONE]\n\n"
                    elif s.startswith("data: "):
                        yield s + "\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            logger.warning(f"[{request_id}] Command Code API streaming error")
            err, done = yield_sse_error(model, "[Command Code API streaming error]")
            yield err; yield done
    finally:
        try:
            if resp_cm is not None:
                await resp_cm.__aexit__(None, None, None)
        except Exception:
            pass
        if owns_session:
            try:
                await session.close()
            except Exception:
                pass
