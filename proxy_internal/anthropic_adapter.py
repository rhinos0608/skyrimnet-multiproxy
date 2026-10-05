import json
import time

import aiohttp
from fastapi import HTTPException


def _combine_system_messages(messages):
    parts = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    if not parts:
        return None
    return "\n\n".join(parts)



def build_anthropic_request(messages, model, max_tokens, system_prompt=None, **extra_params):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [],
    }

    system = system_prompt or _combine_system_messages(messages)
    if system:
        payload["system"] = system

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            continue
        payload["messages"].append(message)

    tools = extra_params.get("tools")
    if tools:
        anthropic_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            anthropic_tools.append({
                "name": name,
                "description": function.get("description"),
                "input_schema": function.get("parameters") or {"type": "object"},
            })
        if anthropic_tools:
            payload["tools"] = anthropic_tools

    tool_choice = extra_params.get("tool_choice")
    if tool_choice == "auto":
        payload["tool_choice"] = {"type": "auto"}
    elif tool_choice == "required":
        payload["tool_choice"] = {"type": "any"}
    elif tool_choice == "none":
        payload["tool_choice"] = {"type": "none"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                payload["tool_choice"] = {"type": "tool", "name": name}
    elif isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "any", "none"}:
            payload["tool_choice"] = {"type": choice_type}
        elif choice_type == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                payload["tool_choice"] = {"type": "tool", "name": name}

    return payload



def normalize_anthropic_error(data):
    error = (data or {}).get("error") or {}
    error_type = error.get("type") or "error"
    return {
        "error": {
            "message": error.get("message", "Unknown error"),
            "type": error_type,
            "param": None,
            "code": error_type,
        }
    }



def normalize_anthropic_response(model, anthropic_data):
    content_parts = []
    tool_calls = []
    for block in (anthropic_data or {}).get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tc_index = len(tool_calls)
            input_data = block.get("input") or {}
            input_json = json.dumps(input_data, separators=(",", ":")) if input_data else "{}"
            tool_calls.append({
                "index": tc_index,
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": input_json,
                },
            })

    usage = (anthropic_data or {}).get("usage") or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    finish_reason = "stop"
    stop_reason = (anthropic_data or {}).get("stop_reason")
    if stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "tool_use":
        finish_reason = "tool_calls"

    message = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": (anthropic_data or {}).get("id") or f"chatcmpl-{int(time.time())}",
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


async def translate_anthropic_sse_events(model, events):
    del model

    tool_buffers = {}
    finish_reason = None

    async for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "content_block_start":
            content_block = event.get("content_block") or {}
            if content_block.get("type") == "tool_use":
                initial_input = content_block.get("input")
                if not initial_input:
                    input_json = ""
                else:
                    input_json = json.dumps(initial_input, separators=(",", ":"))
                tool_buffers[event.get("index")] = {
                    "id": content_block.get("id"),
                    "name": content_block.get("name"),
                    "input_json": input_json,
                }
            continue

        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
                continue
            if delta_type == "input_json_delta":
                buffer = tool_buffers.get(event.get("index"))
                if buffer is not None:
                    buffer["input_json"] += delta.get("partial_json", "")
            continue

        if event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason == "tool_use":
                finish_reason = "tool_calls"
            elif stop_reason in ("end_turn", "stop_sequence", "refusal", "pause_turn"):
                finish_reason = "stop"
            elif stop_reason == "max_tokens":
                finish_reason = "length"
            continue

        if event_type == "message_stop":
            if tool_buffers:
                tool_calls = []
                for tool_index, block_index in enumerate(sorted(tool_buffers)):
                    buffer = tool_buffers[block_index]
                    input_json = buffer.get("input_json", "")
                    if input_json == "":
                        input_json = "{}"
                    try:
                        json.loads(input_json)
                    except Exception as exc:
                        raise ValueError("incomplete tool JSON") from exc
                    tool_calls.append({
                        "index": tool_index,
                        "id": buffer.get("id"),
                        "type": "function",
                        "function": {
                            "name": buffer.get("name"),
                            "arguments": input_json,
                        },
                    })
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"

            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason or "stop",
                }],
            }
            yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"

            yield "data: [DONE]\n\n"


async def call_anthropic_adapter_direct(system_prompt, messages, model, max_tokens, *, api_key, base_url, session=None, auth_headers=None, **extra_params):

    payload = build_anthropic_request(messages, model, max_tokens, system_prompt=system_prompt, **extra_params)
    endpoint = f"{base_url}/v1/messages"
    if auth_headers:
        headers = auth_headers
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    owns_session = session is None
    if owns_session:
        import proxy
        session = proxy.create_session()

    try:
        async with session.post(endpoint, headers=headers, json=payload) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise HTTPException(status_code=resp.status, detail=normalize_anthropic_error(data))
            normalized = normalize_anthropic_response(model, data)
            choices = normalized.get("choices") or []
            if not choices:
                raise HTTPException(status_code=502, detail="Anthropic adapter returned no choices")
            return (choices[0].get("message") or {}).get("content") or ""
    except aiohttp.ClientError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic adapter request failed: {exc}") from exc
    finally:
        if owns_session:
            await session.close()


async def call_anthropic_adapter_streaming(system_prompt, messages, model, max_tokens, *, api_key, base_url, session=None, auth_headers=None, **extra_params):

    payload = build_anthropic_request(messages, model, max_tokens, system_prompt=system_prompt, **extra_params)
    payload["stream"] = True
    endpoint = f"{base_url}/v1/messages"
    if auth_headers:
        headers = auth_headers
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    owns_session = session is None
    if owns_session:
        import proxy
        session = proxy.create_session()

    async def _events():
        try:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    data = await resp.json()
                    raise HTTPException(status_code=resp.status, detail=normalize_anthropic_error(data))

                event_lines = []
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        if not event_lines:
                            continue
                        data_lines = [entry[5:].strip() for entry in event_lines if entry.startswith("data:")]
                        event_lines = []
                        if not data_lines:
                            continue
                        data_str = "\n".join(data_lines)
                        if data_str == "[DONE]":
                            break
                        yield json.loads(data_str)
                        continue
                    event_lines.append(line)

                if event_lines:
                    data_lines = [entry[5:].strip() for entry in event_lines if entry.startswith("data:")]
                    if data_lines:
                        data_str = "\n".join(data_lines)
                        if data_str != "[DONE]":
                            yield json.loads(data_str)
        except aiohttp.ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Anthropic adapter stream failed: {exc}") from exc
        finally:
            if owns_session:
                await session.close()

    async for chunk in translate_anthropic_sse_events(model, _events()):
        yield chunk
