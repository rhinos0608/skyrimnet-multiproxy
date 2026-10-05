"""NVIDIA NIM OpenAI-compatible provider.

Supports thinking/thinking mode via chat_template_kwargs injection per model:
- DeepSeek V3.x, R1 distills: { thinking: true }
- GLM-5, GLM-4.7: { enable_thinking: true, clear_thinking: false }
- Kimi K2.5, K2-thinking: { thinking: true }
- Qwen3, QwQ: { enable_thinking: true }
- MiniMax: uses inline <think> tags rather than reasoning_content field

Maps thinking levels: "minimal" → "low" (NIM only accepts low/medium/high).
Uses system role instead of developer (developer + chat_template_kwargs causes 500 errors).
"""

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
    _model_uses_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

_NVIDIA_MODEL_ALIASES: dict[str, str] = {
    # DeepSeek (thinking: true)
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
    "deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "deepseek-v3.1-terminus": "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek-v4.1-flash": "deepseek-ai/deepseek-v4.1-flash",
    "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
    # DeepSeek R1 distills (thinking: true)
    "deepseek-r1": "deepseek-ai/deepseek-r1",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-70b": "deepseek-ai/deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b": "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-qwen-14b": "deepseek-ai/deepseek-r1-distill-qwen-14b",
    # DeepSeek Coder
    "deepseek-coder-v2-lite": "deepseek-ai/deepseek-coder-v2-lite-instruct",
    # Moonshot AI / Kimi (thinking: true)
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "kimi-k3": "moonshotai/kimi-k3",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking",
    "kimi-k2": "moonshotai/kimi-k2-instruct",
    "kimi-k2-0905": "moonshotai/kimi-k2-instruct-0905",
    # MiniMax (inline <think> tags)
    "minimax-m2": "minimaxai/minimax-m2",
    "minimax-m2.1": "minimaxai/minimax-m2.1",
    "minimax-m2.5": "minimaxai/minimax-m2.5",
    "minimax-m2.7": "minimaxai/minimax-m2.7",
    # Z.AI / GLM (enable_thinking: true)
    "glm-5": "z-ai/glm5",
    "glm-5.3": "z-ai/glm-5.3",
    "glm-5.1": "z-ai/glm-5.1",
    "glm4.7": "z-ai/glm4.7",
    "glm-4.7": "z-ai/glm4.7",
    # OpenAI OSS
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    # Qwen (enable_thinking: true)
    "qwen3": "qwen/qwen3-32b",
    "qwen3-coder-480b": "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen3-next-80b": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen3-next-thinking": "qwen/qwen3-next-80b-a3b-thinking",
    "qwen3-coder-next": "qwen/qwen3-coder-next",
    "qwen2.5-coder-32b": "qwen/qwen2.5-coder-32b-instruct",
    "qwq-32b": "qwen/qwq-32b",
    # Meta Llama
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta/llama-3.1-405b-instruct",
    "llama-3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.2-11b-vision": "meta/llama-3.2-11b-vision-instruct",
    "llama-3.2-1b": "meta/llama-3.2-1b-instruct",
    "llama-3.2-3b": "meta/llama-3.2-3b-instruct",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-4-maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "llama-4-scout": "meta/llama-4-scout-17b-128e-instruct",
    # Code Llama
    "codellama-13b": "meta/codellama-13b-instruct",
    "codellama-34b": "meta/codellama-34b-instruct",
    "codellama-70b": "meta/codellama-70b-instruct",
    # Mistral AI
    "mistral-7b": "mistralai/mistral-7b-instruct-v0.3",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1",
    "mistral-small-3.1": "mistralai/mistral-small-3.1-24b-instruct-2503",
    "mistral-small-24b": "mistralai/mistral-small-24b-instruct-2501",
    "mistral-medium-3": "mistralai/mistral-medium-3-instruct",
    "mistral-large-3": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-small-4": "mistralai/mistral-small-4-119b-2603",
    "mistral-nemotron": "mistralai/mistral-nemotron",
    "devstral-2": "mistralai/devstral-2-123b-instruct-2512",
    "magistral-small": "mistralai/magistral-small-2506",
    "mistral-nemo-12b": "nv-mistralai/mistral-nemo-12b-instruct",
    "mistral-nemo-minitron-8b": "nv-mistralai/mistral-nemo-minitron-8b-8k-instruct",
    # NVIDIA
    "nemotron-4-340b": "nvidia/nemotron-4-340b-instruct",
    "nemotron-3.5-lightning": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron-3-nano": "nvidia/nemotron-3-nano",
    "nemotron-3-nano-30b": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-3-super-120b": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nemotron-super-49b": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nemotron-super-49b-v1.5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-nano-9b-v2": "nvidia/nvidia-nemotron-nano-9b-v2",
    "llama-3.1-nemotron-nano-8b": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    # Microsoft
    "phi-4-mini": "microsoft/phi-4-mini-instruct",
    "phi-3-mini": "microsoft/phi-3-mini-4k-instruct",
    "phi-3-medium": "microsoft/phi-3-medium-128k-instruct",
    "phi-3.5-mini": "microsoft/phi-3.5-mini-instruct",
    # IBM
    "granite-3.3-8b": "ibm/granite-3.3-8b-instruct",
    # Google
    "gemma-2-2b": "google/gemma-2-2b-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "gemma-3n-e4b": "google/gemma-3n-e4b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    # StarCoder
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-15b": "bigcode/starcoderbase-15b",
    # StepFun
    "step-3.5-flash": "stepfun-ai/step-3.5-flash",
    "step-3.7-flash": "stepfun-ai/step-3.7-flash",
    # ByteDance
    "seed-oss-36b": "bytedance/seed-oss-36b-instruct",
    # Cohere
    "command-a": "cohere/command-a",
    # Sarvam
    "sarvam-m": "sarvamai/sarvam-m",
    # Stockmark
    "stockmark-2-100b": "stockmark/stockmark-2-100b-instruct",
}


def _resolve_nvidia_model(model: str) -> str:
    """Strip the 'nvidia:' prefix and expand short aliases to full API paths."""
    raw = model[len("nvidia:") :]
    return _NVIDIA_MODEL_ALIASES.get(raw, raw)


# --- Thinking support per model family ---
# NVIDIA NIM uses chat_template_kwargs to enable thinking per-model.

_NVIDIA_THINKING_KWARGS: dict[str, dict] = {
    # DeepSeek V3.x, R1 distills: { thinking: true }
    "deepseek-v3": {"thinking": True},
    "deepseek-r1": {"thinking": True},
    "deepseek-r1-distill": {"thinking": True},
    # Kimi K2.5, K2-thinking, K3: { thinking: true }
    "kimi-k2": {"thinking": True},
    "kimi-k3": {"thinking": True},
    # GLM-5, GLM-4.7: { enable_thinking: true, clear_thinking: false }
    "glm5": {"enable_thinking": True, "clear_thinking": False},
    "glm-5": {"enable_thinking": True, "clear_thinking": False},
    "glm-4.7": {"enable_thinking": True, "clear_thinking": False},
    # Qwen3, QwQ: { enable_thinking: true }
    "qwen3": {"enable_thinking": True},
    "qwq": {"enable_thinking": True},
}


# Models that enable thinking by default (need explicit disable when thinking is off)
_NVIDIA_THINKING_BY_DEFAULT: set[str] = {
    "glm5",
    "glm-4.7",
}


# Maps "minimal" -> "low" (NIM only accepts low/medium/high)
_THINKING_LEVEL_MAP = {
    "minimal": "low",
}


def _get_thinking_kwargs(model: str, thinking_level: Optional[str]) -> Optional[dict]:
    """Get chat_template_kwargs to inject for thinking-enabled models.

    Args:
        model: The resolved model name (e.g., "deepseek-ai/deepseek-v3.2")
        thinking_level: One of None (off), "minimal", "low", "medium", "high"

    Returns:
        Optional dict with thinking kwargs to inject, or None if no thinking support needed.
    """
    # Extract the base model identifier for matching
    model_part = model.split("/")[-1] if "/" in model else model

    # Check if the model family supports thinking kwargs
    thinking_kwargs = None
    for family, kwargs in _NVIDIA_THINKING_KWARGS.items():
        if family in model_part:
            thinking_kwargs = dict(kwargs)  # Copy for modification
            break

    if thinking_kwargs is None:
        return None

    # Map the thinking level
    level = thinking_level
    if level:
        level = _THINKING_LEVEL_MAP.get(level, level)

    # Handle off / disabled
    if (
        level is None
        or level == "off"
        or (isinstance(level, dict) and level.get("type") == "disabled")
    ):
        # For models that think by default, we need explicit disable
        if model_part in _NVIDIA_THINKING_BY_DEFAULT:
            if "enable_thinking" in thinking_kwargs:
                return {"enable_thinking": False}
            if "thinking" in thinking_kwargs:
                return {"thinking": False}
        return None

    return thinking_kwargs


_NVIDIA_UNSUPPORTED_PARAMS = {
    "reasoning",
    "thinking",
    "top_k",
    "provider",
    "chat_template_kwargs",
}


def _nvidia_payload_fixup(
    payload: dict,
    extra_params: dict,
    model: str,
    thinking_level: Optional[str] = None,
) -> None:
    """Apply NVIDIA NIM-specific payload adjustments.

    - Strips unsupported params (reasoning, thinking, top_k, provider).
    - Injects chat_template_kwargs for thinking-enabled models.
    - Maps "minimal" thinking level to "low".
    - Uses system role instead of developer (developer + chat_template_kwargs = 500 error).
    """
    # Extract thinking level from extra_params if not provided
    if thinking_level is None:
        thinking = extra_params.get("thinking")
        reasoning_effort = extra_params.get("reasoning_effort")

        if thinking is not None and isinstance(thinking, dict):
            # Extract the thinking type/budget from Anthropic-style thinking dict
            if thinking.get("type") == "enabled":
                budget = thinking.get("budget_tokens", 0)
                if budget <= 2048:
                    thinking_level = "low"
                elif budget <= 8192:
                    thinking_level = "medium"
                else:
                    thinking_level = "high"
            elif thinking.get("type") == "disabled":
                thinking_level = "off"
        elif reasoning_effort is not None:
            # Use OAI-style reasoning_effort
            thinking_level = str(reasoning_effort)

    # Inject thinking kwargs if the model supports it
    api_model = payload.get("model", "")
    thinking_kwargs = _get_thinking_kwargs(api_model, thinking_level)
    if thinking_kwargs:
        # chat_template_kwargs must be at the top level
        payload["chat_template_kwargs"] = thinking_kwargs

    # Inject the remaining supported extra params
    payload.update(
        {
            k: v
            for k, v in extra_params.items()
            if v is not None and k not in _NVIDIA_UNSUPPORTED_PARAMS
        }
    )

    # Remove reasoning_effort if we injected chat_template_kwargs
    # (they conflict and can cause errors)
    if thinking_kwargs:
        payload.pop("reasoning_effort", None)


async def call_nvidia_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
) -> str:
    """Forward request to NVIDIA NIM (OpenAI-compatible), collect full response."""
    import proxy

    if not proxy.nvidia_api_key:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA NIM API key not configured — add it via /config/nvidia-key",
        )

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{proxy.NVIDIA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {proxy.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    # Use system role instead of developer to avoid 500 errors with chat_template_kwargs
    use_system = not _model_uses_oai_messages(model)
    payload = {
        "model": api_model,
        "messages": build_oai_messages(system_prompt, messages)
        if use_system
        else messages,
        "max_tokens": max_tokens,
    }
    _nvidia_payload_fixup(payload, extra_params, model)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = (
        session is not proxy.third_party_session and session is not proxy.auth.session
    )

    async def _do_call() -> str:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status in (401, 403):
                raise HTTPException(
                    status_code=401, detail="NVIDIA NIM auth failed — check API key"
                )
            if resp.status == 429:
                raise HTTPException(
                    status_code=429, detail="NVIDIA NIM rate limit exceeded"
                )
            if resp.status == 529:
                await resp.read()
                logger.warning(
                    f"[{request_id}] NVIDIA NIM overloaded (529) — returning empty completion"
                )
                return ""
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(
                    f"[{request_id}] NVIDIA NIM {resp.status}: {error_body[:300]}"
                )
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(
                    f"[{request_id}] NVIDIA NIM returned empty choices array"
                )
                raise HTTPException(
                    status_code=500, detail="NVIDIA NIM returned empty response"
                )
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                if _is_reasoning_truncated(data):
                    logger.info(
                        f"[{request_id}] NVIDIA NIM reasoning truncated, retrying without max_tokens"
                    )
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    async with session.post(
                        endpoint, json=retry_payload, headers=headers
                    ) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(
                                retry_data.get("choices", [{}])[0].get("message", {})
                            )
                            if content:
                                return content
                logger.warning(
                    f"[{request_id}] NVIDIA NIM returned null content, full response: {str(data)[:500]}"
                )
                raise HTTPException(
                    status_code=500, detail="NVIDIA NIM returned empty content"
                )
            return content

    try:
        try:
            content = await proxy._with_retry(
                _do_call, operation=f"NVIDIA NIM {api_model}", request_id=request_id
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="NVIDIA NIM request timed out")
        elapsed = time.time() - start
        logger.info(
            f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, NVIDIA NIM)"
        )
        return content
    finally:
        if owns_session:
            await session.close()


async def call_nvidia_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **extra_params,
):
    """Forward request to NVIDIA NIM with streaming, passthrough SSE directly."""
    import proxy

    if not proxy.nvidia_api_key:
        err, done = yield_sse_error(model, "[NVIDIA NIM Error: API key not configured]")
        yield err
        yield done
        return

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{proxy.NVIDIA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {proxy.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    # Use system role instead of developer to avoid 500 errors with chat_template_kwargs
    use_system = not _model_uses_oai_messages(model)
    payload = {
        "model": api_model,
        "messages": build_oai_messages(system_prompt, messages)
        if use_system
        else messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    _nvidia_payload_fixup(payload, extra_params, model)

    request_id = uuid.uuid4().hex[:8]
    logger.info(
        f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs, stream)"
    )
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = (
        session is not proxy.third_party_session and session is not proxy.auth.session
    )
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                endpoint,
                json_payload=payload,
                headers=headers,
                operation="NVIDIA NIM",
                request_id=request_id,
            )
        except HTTPException as e:
            if e.status_code == 529:
                logger.warning(
                    f"[{request_id}] NVIDIA NIM overloaded (529) — emitting empty completion"
                )
                for chunk in proxy._synthetic_stop_done(model, needs_role=True):
                    yield chunk
                return
            err, done = yield_sse_error(model, f"[NVIDIA NIM Error {e.status_code}]")
            yield err
            yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(
                resp, request_id, "NVIDIA NIM", start
            ):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event

            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] NVIDIA NIM streaming error: {e}")
            err, done = yield_sse_error(model, f"[NVIDIA NIM Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
