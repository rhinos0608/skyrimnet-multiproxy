"""Codex CLI (OpenAI) provider — spawns the local Codex CLI subprocess."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from typing import Optional

from fastapi import HTTPException

import aiohttp

from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

def _extract_codex_error_from_output(raw_output: str) -> Optional[str]:
    """Extract error message from Codex JSONL stdout output.
    
    The Codex CLI returns errors as JSONL in stdout, not stderr.
    Checks for: error types, turn.failed, item.error.
    """
    for line in raw_output.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            # Direct error field
            if "error" in event:
                err = event["error"]
                if isinstance(err, dict):
                    return err.get("message", str(err))
                return str(err)
            # turn.failed wrapper
            msg = event.get("msg", {})
            if isinstance(msg, dict):
                if msg.get("type") == "turn.failed":
                    inner = msg.get("error", {})
                    return inner.get("message", str(inner))
            # Legacy item.error format
            item = event.get("item", {})
            if item.get("type") == "error":
                return item.get("message", "Unknown error")
        except json.JSONDecodeError:
            continue
    return None



def _get_codex_command() -> tuple[str, list[str]]:
    """
    Get the correct command to run Codex CLI on the current platform.

    On Windows, .cmd files need to be executed through cmd.exe.
    Returns (executable, prefix_args) where prefix_args are additional args
    that go before the codex arguments.
    """
    import proxy
    if not proxy.CODEX_PATH:
        return ("", [])

    # On Windows, .cmd/.bat files must be run through cmd.exe
    if sys.platform == "win32" and proxy.CODEX_PATH.lower().endswith((".cmd", ".bat")):
        # Use full path to cmd.exe to ensure it's found regardless of cwd
        cmd_exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        return (cmd_exe, ["/c", proxy.CODEX_PATH])

    return (proxy.CODEX_PATH, [])


def _convert_messages_to_codex_input(messages: list) -> list:
    """Convert OpenAI-format messages to Codex Responses API input format."""
    input_items = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            # System messages go in instructions field, not input
            continue
        elif role == "user":
            input_items.append({"role": "user", "content": content})
        elif role == "assistant":
            input_items.append({"role": "assistant", "content": content})
    return input_items


def _create_isolated_codex_home() -> tuple[str, dict]:
    """
    Create an isolated HOME directory with minimal Codex config.
    Returns (isolated_home_path, env_dict).
    """
    import proxy
    # Create isolated home directory
    isolated_home = tempfile.mkdtemp(prefix="codex_isolated_")
    codex_dir = os.path.join(isolated_home, ".codex")
    os.makedirs(codex_dir, exist_ok=True)

    # Write minimal config.toml (no instructions, no skills)
    config_content = f'''# Minimal isolated Codex config
model = "{proxy.DEFAULT_CODEX_MODEL}"

[approval]
mode = "suggest"

[features]
# Disable all optional features for clean isolation
'''
    with open(os.path.join(codex_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write(config_content)

    # Copy auth.json from real home if it exists
    real_auth_path = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(real_auth_path):
        import shutil
        shutil.copy2(real_auth_path, os.path.join(codex_dir, "auth.json"))

    # Build isolated environment
    env = os.environ.copy()
    env["HOME"] = isolated_home
    env["USERPROFILE"] = isolated_home  # Windows
    env["CODEX_HOME"] = codex_dir

    return isolated_home, env


def _cleanup_isolated_home(isolated_home: str) -> None:
    """Clean up the isolated HOME directory after use."""
    import shutil
    try:
        shutil.rmtree(isolated_home, ignore_errors=True)
    except Exception:
        pass


def _build_codex_exec_args(model: str) -> tuple[str, ...]:
    """Return the Codex CLI argv with fast-mode overrides.

    Reasoning effort is resolved via ``proxy._active_codex_reasoning_effort()``
    so the dashboard reasoning override can lift Codex out of fast-mode without
    re-plumbing kwargs through ``call_codex_streaming``.
    """
    import proxy
    effort = proxy._active_codex_reasoning_effort()
    return (
        "exec",
        "--model", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-",
    )


async def _call_codex_direct_cli(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Spawn Codex CLI subprocess with isolated HOME (clean config, no global instructions)."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.CODEX_PATH:
        raise HTTPException(status_code=503, detail="Codex CLI not installed")

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated)")
    start = time.time()

    isolated_home = None
    proc = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        logger.info(f"[{request_id}] Executing: {executable} {' '.join(prefix_args)} exec --model {model}")
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            *_build_codex_exec_args(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,  # Run from isolated directory
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")), timeout=180
        )
        elapsed = time.time() - start

        raw_output = stdout.decode("utf-8", errors="replace").strip()

        # Check for errors in stdout (JSONL) or non-zero exit code
        stdout_error = _extract_codex_error_from_output(raw_output)
        if proc.returncode != 0 or stdout_error:
            stderr_text = stderr.decode("utf-8", errors="replace")
            error_detail = stdout_error or stderr_text or f"Codex CLI exited with code {proc.returncode}"
            logger.error(f"[{request_id}] Codex CLI failed: {error_detail[:300]}")
            raise HTTPException(status_code=500, detail=f"Codex CLI error: {error_detail[:200]}")

        # Parse JSONL output and extract agent_message text
        response_text = ""
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        for line in raw_output.split("\n"):
            if line.strip():
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Current Codex CLI format: {"id": ..., "msg": {"type": "agent_message", "message": "..."}}
                    msg = event.get("msg", {})
                    if isinstance(msg, dict) and msg.get("type") == "agent_message":
                        text = msg.get("message", "")
                        if text:
                            response_text = text

                    # Legacy format: {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
                    elif event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                    elif event_type == "message.completed":
                        for content in event.get("message", {}).get("content", []):
                            if content.get("type") == "output_text":
                                response_text = content.get("text", "")
                except json.JSONDecodeError:
                    pass

        if not response_text:
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in output, raw length={len(raw_output)}")
            response_text = raw_output

        logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s)")
        return response_text
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        raise HTTPException(status_code=504, detail="Codex CLI timeout")
    finally:
        if proc is not None:
            try:
                if proc.returncode is None:
                    await proc.wait()
            except ProcessLookupError:
                pass
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


_CODEX_TIMEOUT = 180.0
_CODEX_EXIT_GRACE = 5.0


def _extract_codex_line_text(line: str) -> Optional[str]:
    """Extract agent-message text from one Codex JSONL stdout line.

    Supports all three shapes seen in the wild:
      - nested ``msg.type == "agent_message"``
      - ``item.completed`` with ``item.type == "agent_message"``
      - ``message.completed`` with ``content[].type == "output_text"``
    Returns None for lines that carry no agent text.
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return _extract_codex_line_text_from_event(event)


def _extract_codex_line_text_from_event(event) -> Optional[str]:
    """Extract agent-message text from a decoded Codex JSONL event dict."""
    if not isinstance(event, dict):
        return None
    event_type = event.get("type", "")

    msg = event.get("msg", {})
    if isinstance(msg, dict) and msg.get("type") == "agent_message":
        text = msg.get("message", "")
        return text or None

    if event_type == "item.completed":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            return item.get("text") or None

    if event_type == "message.completed":
        message = event.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, list):
            return None
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text":
                text = item.get("text", "")
                if text:
                    return text
    return None


def _extract_codex_line_error(line: str) -> Optional[str]:
    """Extract an error message from a single Codex JSONL stdout line.

    Lets streaming abort early when the CLI emits an error event instead of
    waiting (bounded) for a process that is already failing or hung.
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return _extract_codex_line_error_from_event(event)


def _extract_codex_line_error_from_event(event) -> Optional[str]:
    """Extract an error message from a decoded Codex JSONL event dict."""
    if not isinstance(event, dict):
        return None
    if event.get("error"):
        err = event["error"]
        if isinstance(err, dict):
            return err.get("message", str(err))
        return str(err)
    msg = event.get("msg", {})
    if isinstance(msg, dict) and msg.get("type") == "turn.failed":
        inner = msg.get("error")
        if isinstance(inner, dict):
            return inner.get("message", str(inner))
        return str(inner)
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "error":
        return item.get("message", "Unknown error")
    return None


def _emit_codex_text_as_oai(text: str, cmpl_id: str, created: int, model: str) -> list[str]:
    """Split agent text into 20-char OAI SSE content chunks."""
    chunks = []
    chunk_size = 20  # characters per chunk
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        oai_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(oai_chunk)}\n\n")
    return chunks


async def _call_codex_streaming_cli(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Spawn Codex CLI subprocess and yield output as SSE stream.

    The prompt is fed through stdin; stdout is consumed incrementally as
    JSONL (one line at a time) and each supported agent-message text is
    emitted as OAI SSE as soon as its line arrives — the first content does
    not wait for process exit. stderr is drained concurrently so a chatty
    CLI cannot deadlock stdout processing. The whole operation is bounded by
    ``_CODEX_TIMEOUT``; on timeout, cancellation, or consumer ``aclose`` the
    child is killed and reaped (never awaited forever).
    """
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.CODEX_PATH:
        error, done = yield_sse_error(
            model, "[Codex CLI error: not installed]", status_code=503
        )
        yield error
        yield done
        return

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated stream)")
    start = time.time()

    # Send initial chunk with role
    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }

    isolated_home = None
    proc = None
    stdin_task = None
    stderr_task = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            *_build_codex_exec_args(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,
        )
        yield f"data: {json.dumps(role_chunk)}\n\n"

        emitted_text = ""
        stdout_text = ""
        stderr_text = ""
        early_error = None
        try:
            async with asyncio.timeout(_CODEX_TIMEOUT):
                # Concurrent stderr drain: reading stdout and stderr in
                # separate coroutines prevents pipe-buffer deadlocks.
                async def _drain_stderr():
                    nonlocal stderr_text
                    data = await proc.stderr.read()
                    stderr_text = data.decode("utf-8", errors="replace")

                stderr_task = asyncio.create_task(_drain_stderr())

                async def _submit_stdin():
                    try:
                        proc.stdin.write(full_prompt.encode("utf-8"))
                        await proc.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass

                stdin_task = asyncio.create_task(_submit_stdin())

                # Consume stdout JSONL incrementally, line by line.
                line_buffer = bytearray()
                async for raw in proc.stdout:
                    line_buffer.extend(raw)
                    while True:
                        idx = line_buffer.find(b"\n")
                        if idx < 0:
                            break
                        line = bytes(line_buffer[:idx]).strip()
                        del line_buffer[:idx + 1]
                        if not line:
                            continue
                        line_text = line.decode("utf-8", errors="replace")
                        stdout_text += line_text + "\n"
                        try:
                            event = json.loads(line_text)
                        except json.JSONDecodeError:
                            continue
                        line_error = _extract_codex_line_error_from_event(event)
                        if line_error:
                            early_error = line_error
                            break  # CLI is failing/hanging — stop consuming
                        text = _extract_codex_line_text_from_event(event)
                        if text:
                            emitted_text += text
                            for chunk in _emit_codex_text_as_oai(
                                text, cmpl_id, created, model
                            ):
                                yield chunk
                    if early_error:
                        break
                if line_buffer.strip() and not early_error:
                    line_text = line_buffer.decode("utf-8", errors="replace").strip()
                    stdout_text += line_text + "\n"
                    try:
                        event = json.loads(line_text)
                    except json.JSONDecodeError:
                        event = None
                    if event is not None:
                        line_error = _extract_codex_line_error_from_event(event)
                        if line_error:
                            early_error = line_error
                        else:
                            text = _extract_codex_line_text_from_event(event)
                        if text:
                            emitted_text += text
                            for chunk in _emit_codex_text_as_oai(
                                text, cmpl_id, created, model
                            ):
                                yield chunk

                if stdin_task is not None and not stdin_task.done():
                    stdin_task.cancel()
                    try:
                        await stdin_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if stderr_task is not None and not stderr_task.done():
                    # Give stderr a short window to finish naturally so
                    # stderr_text is populated before we cancel.
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(stderr_task), timeout=0.5
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    if not stderr_task.done():
                        stderr_task.cancel()
                    try:
                        await stderr_task
                    except (asyncio.CancelledError, Exception):
                        pass

                # Bounded wait for exit; if the child never exits (early error
                # abort, leaked fd), kill and reap it — never wait forever.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_CODEX_EXIT_GRACE)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=_CODEX_EXIT_GRACE)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        pass
        except asyncio.TimeoutError:
            raise

        elapsed = time.time() - start

        # Preserve stdout error extraction: prefer a JSONL error from stdout,
        # fall back to stderr text, then the exit code.
        stdout_error = _extract_codex_error_from_output(stdout_text) or early_error
        if proc.returncode != 0 or stdout_error:
            err_detail = stdout_error or stderr_text or f"Codex CLI exited with code {proc.returncode}"
            logger.error(f"[{request_id}] Codex CLI failed: {err_detail[:300]}")
            err_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"[Codex error: {err_detail[:100]}]"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if not emitted_text:
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in output, raw length={len(stdout_text)}")
            emitted_text = stdout_text
            for chunk in _emit_codex_text_as_oai(emitted_text, cmpl_id, created, model):
                yield chunk

        # Final chunk
        stop_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"[{request_id}] <- {len(emitted_text)} chars ({elapsed:.1f}s, incremental stream)")
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        err, done = yield_sse_error(model, "[Codex CLI timeout]")
        yield err; yield done
    except Exception as e:
        logger.error(f"[{request_id}] Codex CLI error: {e}")
        err, done = yield_sse_error(model, f"[Codex CLI error: {e}]")
        yield err; yield done
    finally:
        if stdin_task is not None and not stdin_task.done():
            stdin_task.cancel()
            try:
                await stdin_task
            except (asyncio.CancelledError, Exception):
                pass
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        if proc is not None:
            try:
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=_CODEX_EXIT_GRACE)
                    except (ProcessLookupError, asyncio.TimeoutError):
                        pass
            except Exception:
                pass
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


def _tools_to_system_hint(tools: list) -> str:
    """Describe caller-supplied tools in plain text for the Codex CLI system prompt."""
    if not tools:
        return ""
    lines = ["Available tools (use as needed to complete the task):"]
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        desc = t.get("description", "")
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "\n".join(lines)


async def call_codex_messages_passthrough(body: dict, is_stream: bool):
    """Route an Anthropic /v1/messages body through the Codex CLI subprocess.

    Folds tool definitions into the system prompt so the Codex agent has
    full context, then wraps the CLI output as a native Anthropic message
    response.  The CLI runs its own internal agent loop (bash, file ops, etc.)
    and returns the final answer — callers do not dispatch tools themselves.
    """
    from fastapi.responses import StreamingResponse
    import proxy

    model = body.get("model", proxy.DEFAULT_CODEX_MODEL)
    max_tokens = body.get("max_tokens") or 8192
    messages = body.get("messages") or []

    # Build system prompt: merge top-level system + tool hints.
    system_parts: list[str] = []
    raw_system = body.get("system")
    if isinstance(raw_system, str) and raw_system:
        system_parts.append(raw_system)
    elif isinstance(raw_system, list):
        for block in raw_system:
            if isinstance(block, dict) and block.get("type") == "text":
                system_parts.append(block.get("text", ""))

    tools = body.get("tools") or []
    hint = _tools_to_system_hint(tools)
    if hint:
        system_parts.append(hint)

    system_prompt = "\n\n".join(filter(None, system_parts)) or None

    # Flatten any structured content blocks in messages to plain text.
    flat_messages: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
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
                        result_content = " ".join(
                            b.get("text", "") for b in result_content if isinstance(b, dict)
                        )
                    parts.append(f"[tool_result: {result_content}]")
            content = "\n".join(parts)
        flat_messages.append({"role": role, "content": content})

    if is_stream:
        async def _anthropic_stream():
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield "data: {\"type\": \"ping\"}\n\n"

            output_tokens = 0
            async for raw_chunk in call_codex_streaming(system_prompt, flat_messages, model, max_tokens):
                if raw_chunk.startswith("data: ") and not raw_chunk.startswith("data: [DONE]"):
                    try:
                        oai = json.loads(raw_chunk[6:])
                        delta = (oai.get("choices") or [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            output_tokens += len(text) // 4
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': max(1, output_tokens)}})}\n\n"
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(
            _anthropic_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    # Non-streaming: run direct and wrap in Anthropic message shape.
    # Treat CLI failures as error text in the response body (matching how
    # call_codex_streaming embeds errors as text chunks) so callers always
    # get a valid Anthropic message back rather than a 500/503 HTTP error.
    from fastapi import HTTPException as _HTTPException
    try:
        text = await call_codex_direct(system_prompt, flat_messages, model, max_tokens)
    except _HTTPException as exc:
        text = f"[Codex error: {exc.detail}]"
    except Exception as exc:
        text = f"[Codex error: {exc}]"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": sum(len(m.get("content", "")) for m in flat_messages) // 4,
            "output_tokens": len(text) // 4,
        },
    }


# ---------------------------------------------------------------------------
# Direct (non-CLI) Codex transport: Responses API over ChatGPT backend.
#
# Used FIRST by the wrappers below; the `codex exec` subprocess path above
# (now `_call_codex_direct_cli` / `_call_codex_streaming_cli`) is retained
# as the fallback so a Cloudflare/attestation/endpoint change degrades
# instead of breaking. Private undocumented backend — no stability guarantee.
# ---------------------------------------------------------------------------

CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_CODEX_RESPONSES_URL = CHATGPT_CODEX_BASE_URL + "/responses"
_CODEX_DIRECT_TIMEOUT = 120


def _scrubbed_reason(detail: str, limit: int = 200) -> str:
    """Bound a fallback reason for logs; never carry credential values."""
    text = str(detail or "")
    lowered = text.lower()
    for secret_key in ("authorization", "bearer", "access_token", "refresh_token", "account_id"):
        if secret_key in lowered:
            return f"{text[:limit]} [scrubbed]".strip()
    collapsed = " ".join(text.split())
    return collapsed[:limit]


def _build_responses_body(
    model: str,
    system_prompt: Optional[str],
    messages: list,
) -> dict:
    """Build the Responses API body (NOT chat completions)."""
    converted = _convert_messages_to_codex_input(messages)
    input_items: list[dict] = []
    for m in converted:
        role = m.get("role", "user")
        text = m.get("content", "")
        input_items.append({
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        })
    return {
        "model": model,
        "instructions": system_prompt or "",
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
        "stream": True,
        "include": [],
    }


def _parse_responses_sse(raw: str) -> dict:
    """Parse a Responses SSE payload into text/reasoning/usage/error.

    Reasoning deltas are collected separately and never merged into text,
    mirroring the CLI path which surfaces only the final agent message.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: Optional[dict] = None
    error: Optional[str] = None
    events = 0
    parse_errors = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1
            continue
        if not isinstance(evt, dict):
            parse_errors += 1
            continue
        events += 1
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            delta = evt.get("delta", "")
            if isinstance(delta, str) and delta:
                text_parts.append(delta)
        elif etype in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = evt.get("delta", "")
            if isinstance(delta, str) and delta:
                reasoning_parts.append(delta)
        elif etype == "response.completed":
            resp = evt.get("response") or {}
            if isinstance(resp, dict):
                u = resp.get("usage")
                if isinstance(u, dict):
                    usage = {
                        "input_tokens": int(u.get("input_tokens", 0) or 0),
                        "output_tokens": int(u.get("output_tokens", 0) or 0),
                    }
        elif etype == "response.failed":
            resp = evt.get("response") or evt
            err = resp.get("error") if isinstance(resp, dict) else None
            if isinstance(err, dict):
                error = str(err.get("message", err))
            elif err is not None:
                error = str(err)
            else:
                error = "response.failed"
    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "usage": usage,
        "error": error,
        "events": events,
        "parse_errors": parse_errors,
    }


async def _direct_post(headers: dict, body: dict) -> tuple[int, str]:
    """POST to the Responses endpoint. Seam for tests to monkeypatch."""
    timeout = aiohttp.ClientTimeout(total=_CODEX_DIRECT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            CHATGPT_CODEX_RESPONSES_URL, headers=headers, json=body
        ) as resp:
            return resp.status, await resp.text()


async def _call_codex_direct_http(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> tuple[str, Optional[dict]]:
    """Attempt the direct HTTP path. Raises on any non-usable outcome."""
    import proxy

    auth = proxy.codex_auth
    if not await auth.refresh_if_needed():
        raise RuntimeError("direct transport: auth not ready")
    headers = auth.get_chatgpt_headers()
    body = _build_responses_body(model, system_prompt, messages)
    status, raw = await _direct_post(headers, body)
    if status == 400 and "Store must be set to false" in raw:
        raise RuntimeError("direct transport: 400 Store must be set to false")
    if status in (401, 403):
        raise RuntimeError(f"direct transport: upstream {status}")
    if status < 200 or status >= 300:
        raise RuntimeError(f"direct transport: upstream {status}")
    parsed = _parse_responses_sse(raw)
    if parsed["error"]:
        raise RuntimeError(f"direct transport: {parsed['error']}")
    if not parsed["text"].strip():
        raise RuntimeError("direct transport: empty content")
    return parsed["text"], parsed["usage"]


async def call_codex_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Direct-first wrapper; falls back to the `codex exec` CLI path once."""
    request_id = uuid.uuid4().hex[:8]
    try:
        text, _usage = await _call_codex_direct_http(system_prompt, messages, model, max_tokens)
        logger.info(f"[{request_id}] Codex transport=direct model={model}")
        return text
    except Exception as exc:
        reason = _scrubbed_reason(f"{type(exc).__name__}: {exc}")
        logger.warning(f"[{request_id}] Codex direct failed ({reason}); falling back to CLI")
        result = await _call_codex_direct_cli(system_prompt, messages, model, max_tokens)
        logger.info(f"[{request_id}] Codex transport=cli-fallback model={model}")
        return result


async def _stream_codex_direct_http(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Yield OAI chunks from the direct HTTP path. Raises on failure."""
    import proxy

    auth = proxy.codex_auth
    if not await auth.refresh_if_needed():
        raise RuntimeError("direct transport: auth not ready")
    headers = auth.get_chatgpt_headers()
    body = _build_responses_body(model, system_prompt, messages)
    status, raw = await _direct_post(headers, body)
    if status < 200 or status >= 300:
        raise RuntimeError(f"direct transport: upstream {status}")
    parsed = _parse_responses_sse(raw)
    if parsed["error"]:
        raise RuntimeError(f"direct transport: {parsed['error']}")
    if not parsed["text"].strip():
        raise RuntimeError("direct transport: empty content")
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    for chunk in _emit_codex_text_as_oai(parsed["text"], cmpl_id, created, model):
        yield chunk


async def call_codex_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Direct-first streaming wrapper; single fallback to the CLI path."""
    request_id = uuid.uuid4().hex[:8]
    try:
        got_any = False
        async for chunk in _stream_codex_direct_http(system_prompt, messages, model, max_tokens):
            got_any = True
            yield chunk
        if not got_any:
            raise RuntimeError("direct transport: empty content")
        logger.info(f"[{request_id}] Codex transport=direct model={model}")
        return
    except Exception as exc:
        reason = _scrubbed_reason(f"{type(exc).__name__}: {exc}")
        logger.warning(f"[{request_id}] Codex direct stream failed ({reason}); falling back to CLI")
        async for chunk in _call_codex_streaming_cli(system_prompt, messages, model, max_tokens):
            yield chunk
        logger.info(f"[{request_id}] Codex transport=cli-fallback model={model}")
        return
