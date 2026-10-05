"""Google Cloud Gemini Enterprise Agent Platform provider via Gen AI SDK with API key.

Uses the google-genai SDK with an explicit ``genai.Client(vertexai=True, api_key=...)``
so no ``os.environ`` mutation is needed. Clients are pooled per API key and reused
across concurrent requests; a runtime key rotation retires the old generation and
closes it once its last lease exits. ``close_vertex_clients()`` is wired into the
FastAPI lifespan shutdown.

Models use 'vertex:' prefix, e.g. vertex:gemini-2.5-flash.
"""

import asyncio
import itertools
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import HTTPException

from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Client pool
#
# ``_vertex_clients`` maps api key -> list of generations ``(gen, client, leases)``
# where the LAST entry is the active client for that key. On key rotation the
# old key's generations are retired: unleaded clients close immediately, leased
# clients move to ``_vertex_retired`` (by object id) and close when their last
# lease exits — so in-flight requests keep a live client while a new generation
# is built for the new key. All state is mutated under ``_vertex_lock`` (the
# requests are async and share one event loop).
# ---------------------------------------------------------------------------

_vertex_clients: dict[str, list[tuple[int, object, int]]] = {}
_vertex_retired: dict[int, tuple[object, str, int]] = {}
_vertex_generation = itertools.count()
_vertex_lock = asyncio.Lock()


# Model aliases: short name -> SDK model ID.
# Vertex AI requires exact versioned model names; short aliases map to the
# correct SDK IDs.  Any unrecognised name is passed through as-is.
_VERTEX_MODEL_ALIASES: dict[str, str] = {
    # Gemini 3.x GA line (2.x retired; 3 Pro shut down March 2026)
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.7-flash": "gemini-3.7-flash",
    "gemini-3.8-flash": "gemini-3.8-flash",
}


def _resolve_vertex_model(model: str) -> str:
    """Strip the 'vertex:' prefix and expand short aliases to SDK model IDs."""
    raw = model[len("vertex:"):]
    return _VERTEX_MODEL_ALIASES.get(raw, raw)


def _build_contents(messages: list) -> list:
    """Convert OpenAI-format messages to Gemini Content objects.

    System messages are excluded — they're handled via system_instruction.
    """
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            continue
        if role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    return contents


def _build_client(api_key: str):
    """Create a genai Client configured explicitly for Vertex AI with API key
    auth. Every parameter the SDK might otherwise read from the environment is
    passed explicitly, so no ``os.environ`` mutation is required."""
    from google import genai
    from google.genai.types import HttpOptions

    return genai.Client(
        vertexai=True,
        api_key=api_key,
        http_options=HttpOptions(api_version="v1"),
    )


async def _close_client_object(client) -> None:
    """Close both the sync and async genai transports, ignoring errors."""
    try:
        client.close()
    except Exception:
        pass
    try:
        await client.aio.aclose()
    except Exception:
        pass


async def _retire_key_locked(key: str) -> None:
    """Retire every generation of a key (caller holds ``_vertex_lock``).

    Unleaded clients close immediately; leased clients are tracked in
    ``_vertex_retired`` and closed by ``_release_vertex_client`` when their
    last lease exits.
    """
    for _gen, client, leases in _vertex_clients.pop(key, []):
        if leases > 0:
            _vertex_retired[id(client)] = (client, key, leases)
        else:
            await _close_client_object(client)


async def _acquire_vertex_client() -> tuple[object, str]:
    """Return ``(client, key)`` for the currently configured Vertex API key.

    Builds a fresh client on first use or after a rotation, retires stale
    generations of other keys, and records a lease so the client cannot be
    closed while this request is using it.
    """
    import proxy
    async with _vertex_lock:
        key = proxy.vertex_api_key
        if not key:
            raise HTTPException(
                status_code=503,
                detail="Gemini API key not configured — add it via /config/vertex-key",
            )
        for other_key in list(_vertex_clients):
            if other_key != key:
                await _retire_key_locked(other_key)
        generations = _vertex_clients.get(key)
        if not generations:
            client = await asyncio.to_thread(_build_client, key)
            generations = [(next(_vertex_generation), client, 0)]
            _vertex_clients[key] = generations
        gen, client, leases = generations[-1]
        generations[-1] = (gen, client, leases + 1)
        return client, key


async def _release_vertex_client(client, key: str) -> None:
    """Release one lease taken by ``_acquire_vertex_client``.

    If the client was retired (key rotated) while leased, it is closed when
    its last lease exits. A client still in use is never closed.
    """
    async with _vertex_lock:
        cid = id(client)
        retired = _vertex_retired.get(cid)
        if retired is not None:
            retired_client, retired_key, leases = retired
            if leases <= 1:
                _vertex_retired.pop(cid, None)
                await _close_client_object(retired_client)
            else:
                _vertex_retired[cid] = (retired_client, retired_key, leases - 1)
            return
        generations = _vertex_clients.get(key)
        if generations is None:
            return
        for idx, (gen, candidate, leases) in enumerate(generations):
            if candidate is client:
                # Idle (non-retired) clients stay cached for reuse; only
                # retired clients are closed when their last lease exits.
                generations[idx] = (gen, candidate, max(0, leases - 1))
                return


async def vertex_key_rotated() -> None:
    """Retire clients built under the previous API key.

    Called by the config endpoint when the key changes at runtime so rotation
    takes effect immediately even before the next request arrives. Retired
    clients still leased by in-flight requests close once their last lease
    exits; the next request builds a fresh client for the new key.
    """
    import proxy
    async with _vertex_lock:
        active_key = proxy.vertex_api_key or ""
        for other_key in list(_vertex_clients):
            if other_key != active_key:
                await _retire_key_locked(other_key)


async def close_vertex_clients() -> None:
    """Close every Vertex genai client (sync + async transports). Shutdown hook."""
    async with _vertex_lock:
        for key in list(_vertex_clients):
            for _gen, client, _leases in _vertex_clients.pop(key, []):
                await _close_client_object(client)
        for cid in list(_vertex_retired):
            client, _key, _leases = _vertex_retired.pop(cid)
            await _close_client_object(client)


def _build_config(system_prompt: Optional[str], max_tokens: int):
    """Build GenerateContentConfig from proxy parameters."""
    from google.genai.types import GenerateContentConfig

    cfg = GenerateContentConfig(max_output_tokens=max_tokens)
    if system_prompt:
        cfg.system_instruction = system_prompt
    return cfg


async def call_vertex_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
) -> str:
    """Forward request to Gemini Enterprise Agent Platform via Gen AI SDK."""
    import proxy

    if not proxy.vertex_api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key not configured — add it via /config/vertex-key",
        )

    api_model = _resolve_vertex_model(model)
    config = _build_config(system_prompt, max_tokens)
    contents = _build_contents(messages)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Gemini SDK ({api_model}, {len(messages)} msgs)")
    start = time.time()

    client, client_key = await _acquire_vertex_client()
    try:
        response = await client.aio.models.generate_content(
            model=api_model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Gemini SDK error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini SDK error: {e}")
    finally:
        await _release_vertex_client(client, client_key)

    text = response.text or ""
    elapsed = time.time() - start

    if not text and response.candidates:
        finish = response.candidates[0].finish_reason
        if finish and finish.name == "MAX_TOKENS":
            logger.warning(
                f"[{request_id}] Gemini SDK MAX_TOKENS with {max_tokens} tokens "
                f"(thinking models may need higher max_tokens)"
            )

    logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Gemini SDK)")
    return text


async def call_vertex_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
):
    """Forward request to Gemini Enterprise Agent Platform via Gen AI SDK, streaming."""
    import proxy

    if not proxy.vertex_api_key:
        err, done = yield_sse_error(model, "[Gemini SDK Error: API key not configured]")
        yield err; yield done
        return

    api_model = _resolve_vertex_model(model)
    config = _build_config(system_prompt, max_tokens)
    contents = _build_contents(messages)

    request_id = uuid.uuid4().hex[:8]
    logger.info(
        f"[{request_id}] -> Gemini SDK ({api_model}, {len(messages)} msgs, stream)"
    )
    start = time.time()

    client, client_key = await _acquire_vertex_client()
    try:
        async for chunk in await client.aio.models.generate_content_stream(
            model=api_model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                data = json.dumps({
                    "id": f"gemini-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.text},
                    }],
                })
                yield f"data: {data}\n\n"
    except Exception as e:
        logger.error(f"[{request_id}] Gemini SDK streaming error: {e}")
        err, done = yield_sse_error(model, f"[Gemini SDK Error: {e}]")
        yield err; yield done
        return
    finally:
        await _release_vertex_client(client, client_key)

    elapsed = time.time() - start
    logger.info(
        f"[{request_id}] <- stream complete ({elapsed:.1f}s, Gemini SDK)"
    )
    yield "data: [DONE]\n\n"
