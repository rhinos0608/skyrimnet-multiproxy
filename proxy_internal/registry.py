"""Centralized provider registry — single source of truth for all LLM providers."""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    display_name: str
    category: str  # "chat" or "cli"
    detector: Optional[Callable[[str], bool]]  # None = default/fallback
    call_direct: Callable
    call_streaming: Callable
    health_fn: Callable[[], bool]
    default_model: str
    description: str = ""
    accepts_extra: bool = False
    oai_compatible: bool = False
    error_reason: str = ""
    param_name: str = "message"  # "message" for chat, "task" for CLI
    model_prefix: Optional[str] = None  # e.g. "fireworks:*"; None for default bucket


# ---------------------------------------------------------------------------
# Registry population — import detection + call functions from their modules
# ---------------------------------------------------------------------------

import proxy
from proxy_internal.model_detect import (
    is_ollama_model, is_lmstudio_model, is_openrouter_model, is_codex_model,
    is_bytedance_model,
    is_antigravity_model, is_zai_model,
    is_xiaomi_model, is_opencode_model, is_opencode_zen_model,
    is_opencode_go_model, is_qwen_model, is_kilo_model,
    is_fireworks_model, is_nvidia_model, is_command_code_model,
    is_command_code_api_model, is_vertex_model,
)


def _is_claude_healthy() -> bool:
    return proxy.auth.is_ready or proxy.is_claude_auth_ready()

def _is_codex_healthy() -> bool:
    ready = proxy.codex_auth.is_ready or proxy.ensure_codex_discovery()
    return ready and proxy.CODEX_PATH is not None

def _is_openrouter_healthy() -> bool:
    try:
        return proxy._resolve_openrouter_api_key() is not None
    except Exception:
        return proxy.openrouter_api_key is not None

def _is_kilo_healthy() -> bool:
    return proxy.kilo_api_key is not None

def _is_antigravity_healthy() -> bool:
    if any(a.is_ready for a in proxy.antigravity_auth.accounts):
        return True
    try:
        return bool(proxy.ensure_antigravity_discovery())
    except Exception:
        return False

def _is_zai_healthy() -> bool:
    return proxy.zai_api_key is not None

def _is_xiaomi_healthy() -> bool:
    return proxy.xiaomi_api_key is not None

def _is_opencode_zen_healthy() -> bool:
    try:
        _m, _u, key, _p = proxy._resolve_opencode("opencode:probe")
        return key is not None
    except Exception:
        return proxy.opencode_api_key is not None

def _is_opencode_go_healthy() -> bool:
    try:
        _m, _u, key, _p = proxy._resolve_opencode("opencode-go:glm-5.3")
        return key is not None
    except Exception:
        return proxy.opencode_go_api_key is not None or proxy.opencode_api_key is not None

def _is_qwen_healthy() -> bool:
    return proxy.qwen_auth.is_ready

def _is_fireworks_healthy() -> bool:
    return proxy.fireworks_api_key is not None

def _is_nvidia_healthy() -> bool:
    return proxy.nvidia_api_key is not None


def _is_command_code_healthy() -> bool:
    return proxy.COMMAND_CODE_PATH is not None


def _is_command_api_healthy() -> bool:
    resolver = getattr(proxy, "_resolve_command_code_api_key", None)
    if callable(resolver):
        try:
            return resolver() is not None
        except Exception:
            return False
    return getattr(proxy, "command_code_api_key", None) is not None


def _is_bytedance_healthy() -> bool:
    return proxy.bytedance_api_key is not None


def _is_vertex_healthy() -> bool:
    try:
        return proxy._resolve_vertex_api_key() is not None
    except Exception:
        return proxy.vertex_api_key is not None


_ollama_healthy = False

def _is_ollama_healthy() -> bool:
    if _ollama_healthy:
        return True
    try:
        return proxy._resolve_ollama_api_key() is not None
    except Exception:
        return False

def set_ollama_healthy(val: bool):
    global _ollama_healthy
    _ollama_healthy = val


_lmstudio_healthy = False

def _is_lmstudio_healthy() -> bool:
    return _lmstudio_healthy

def set_lmstudio_healthy(val: bool):
    global _lmstudio_healthy
    _lmstudio_healthy = val


REGISTRY: list[ProviderConfig] = [
    ProviderConfig(
        name="ollama", display_name="Ollama", category="chat",
        detector=is_ollama_model,
        call_direct=proxy.call_ollama_direct,
        call_streaming=proxy.call_ollama_streaming,
        health_fn=_is_ollama_healthy,
        default_model="ollama:llama3.1",
        description="Send a chat message to Ollama (local or cloud). Use 'ollama:model' format.",
        accepts_extra=True, oai_compatible=True,
        error_reason="Ollama is not reachable or has no models",
        model_prefix="ollama:*",
    ),
    ProviderConfig(
        name="lmstudio", display_name="LM Studio", category="chat",
        detector=is_lmstudio_model,
        call_direct=proxy.call_lmstudio_direct,
        call_streaming=proxy.call_lmstudio_streaming,
        health_fn=_is_lmstudio_healthy,
        default_model="lmstudio:default",
        description="Send a chat message to LM Studio (local). Use 'lmstudio:model' format.",
        accepts_extra=True, oai_compatible=True,
        error_reason="LM Studio is not running or has no models loaded",
        model_prefix="lmstudio:*",
    ),
    ProviderConfig(
        name="codex_cli", display_name="Codex", category="cli",
        detector=is_codex_model,
        call_direct=proxy.call_codex_direct,
        call_streaming=proxy.call_codex_streaming,
        health_fn=_is_codex_healthy,
        default_model="gpt-6-sol",
        description="Codex CLI (OpenAI) — the workhorse CLI provider. Highest rate limits and strong all-around capability.",
        param_name="task",
        error_reason="not installed or auth not ready",
        model_prefix="gpt-*|codex-*",
    ),
    ProviderConfig(
        name="fireworks", display_name="Fireworks", category="chat",
        detector=is_fireworks_model,
        call_direct=proxy.call_fireworks_direct,
        call_streaming=proxy.call_fireworks_streaming,
        health_fn=_is_fireworks_healthy,
        default_model="fireworks:kimi-k2p5-turbo",
        description="Send a chat message via Fireworks AI. Use 'fireworks:model' format.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="fireworks:*",
    ),
    ProviderConfig(
        name="nvidia", display_name="NVIDIA NIM", category="chat",
        detector=is_nvidia_model,
        call_direct=proxy.call_nvidia_direct,
        call_streaming=proxy.call_nvidia_streaming,
        health_fn=_is_nvidia_healthy,
        default_model="nvidia:deepseek-v3.2",
        description="Send a chat message via NVIDIA NIM. Free-tier models. Use 'nvidia:model' format.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="nvidia:*",
    ),
    ProviderConfig(
        name="vertex", display_name="Vertex AI", category="chat",
        detector=is_vertex_model,
        call_direct=proxy.call_vertex_direct,
        call_streaming=proxy.call_vertex_streaming,
        health_fn=_is_vertex_healthy,
        default_model="vertex:gemini-3.5-flash",
        description="Google Cloud Vertex AI Gemini Enterprise Agent Platform. OpenAI-compatible. Use 'vertex:model' format.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="vertex:*",
    ),
    ProviderConfig(
            name="bytedance", display_name="ByteDance", category="chat",
            detector=is_bytedance_model,
            call_direct=proxy.call_bytedance_direct,
            call_streaming=proxy.call_bytedance_streaming,
            health_fn=_is_bytedance_healthy,
            default_model="bytedance:bytedance-seed-code",
            description="Anthropic-compatible upstream via ByteDance. Use 'bytedance:model' format.",
            accepts_extra=True, oai_compatible=True,
            error_reason="API key not configured",
            model_prefix="bytedance:*",
        ),
        ProviderConfig(
            name="command_code", display_name="Command Code", category="chat",
            detector=is_command_code_model,
            call_direct=proxy.call_command_code_direct,
            call_streaming=proxy.call_command_code_streaming,
            health_fn=_is_command_code_healthy,
            default_model="command_code:deepseek-v4-flash",
            description="Command Code CLI (cmd --print) — local CLI provider. Use 'command_code:model' format.",
            accepts_extra=True, oai_compatible=True,
            error_reason="CLI not installed — install from commandcode.ai",
            model_prefix="command_code:*",
        ),
        ProviderConfig(
            name="command_api", display_name="Command Code API", category="chat",
            detector=is_command_code_api_model,
            call_direct=proxy.call_command_code_api_direct,
            call_streaming=proxy.call_command_code_api_streaming,
            health_fn=_is_command_api_healthy,
            default_model="command_api:deepseek/deepseek-v4-flash",
            description="Command Code Provider API — OpenAI-compatible chat endpoint (https://commandcode.ai/docs/provider). Needs the $10 GOAT plan or higher ($1 Go has no API). Use 'command_api:vendor/model' format.",
            accepts_extra=True, oai_compatible=True,
            error_reason="API key not configured",
            model_prefix="command_api:*",
        ),
        ProviderConfig(
            name="openrouter", display_name="OpenRouter", category="chat",
            detector=is_openrouter_model,
            call_direct=proxy.call_openrouter_direct,
            call_streaming=proxy.call_openrouter_streaming,
            health_fn=_is_openrouter_healthy,
            default_model="anthropic/claude-sonnet-5.5",
            description="Send a chat message via OpenRouter. Access hundreds of models (use provider/model format).",
            accepts_extra=True, oai_compatible=True,
            error_reason="API key not configured",
            model_prefix="*/*",
        ),
    ProviderConfig(
        name="kilo", display_name="Kilo Gateway", category="chat",
        detector=is_kilo_model,
        call_direct=proxy.call_kilo_direct,
        call_streaming=proxy.call_kilo_streaming,
        health_fn=_is_kilo_healthy,
        default_model="kilo:kilo-auto/balanced",
        description="Kilo Gateway — OpenAI-compatible routing layer with Kilo's managed models and Auto tiers.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="kilo:*",
    ),
    ProviderConfig(
        name="antigravity", display_name="Antigravity", category="chat",
        detector=is_antigravity_model,
        call_direct=proxy.call_antigravity_direct,
        call_streaming=proxy.call_antigravity_streaming,
        health_fn=_is_antigravity_healthy,
        default_model="antigravity-gemini-3-flash",
        description="Antigravity (Google IDE API) — supports Gemini models and Claude endpoints.",
        oai_compatible=True,
        error_reason="auth not ready",
        model_prefix="antigravity-*",
    ),
    ProviderConfig(
        name="zai", display_name="Z.AI", category="chat",
        detector=is_zai_model,
        call_direct=proxy.call_zai_direct,
        call_streaming=proxy.call_zai_streaming,
        health_fn=_is_zai_healthy,
        default_model="zai:glm-4.5-air",
        description="Send a chat message to Z.AI.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="zai:*",
    ),
    ProviderConfig(
        name="xiaomi", display_name="Xiaomi", category="chat",
        detector=is_xiaomi_model,
        call_direct=proxy.call_xiaomi_direct,
        call_streaming=proxy.call_xiaomi_streaming,
        health_fn=_is_xiaomi_healthy,
        default_model="xiaomi:mimo-v2.5-pro",
        description="Send a chat message to Xiaomi MiMo models.",
        accepts_extra=True, oai_compatible=True,
        error_reason="API key not configured",
        model_prefix="xiaomi:*",
    ),
    ProviderConfig(
        name="opencode_zen", display_name="OpenCode Zen", category="cli",
        detector=is_opencode_zen_model,
        call_direct=proxy.call_opencode_direct,
        call_streaming=proxy.call_opencode_streaming,
        health_fn=_is_opencode_zen_healthy,
        default_model="opencode:deepseek-v4-flash-free",
        description="Execute an agentic task via OpenCode Zen CLI. Code-focused agent for local development.",
        accepts_extra=True, oai_compatible=True,
        param_name="task",
        error_reason="API key not configured",
        model_prefix="opencode:*",
    ),
    ProviderConfig(
        name="opencode_go", display_name="OpenCode Go", category="cli",
        detector=is_opencode_go_model,
        call_direct=proxy.call_opencode_direct,
        call_streaming=proxy.call_opencode_streaming,
        health_fn=_is_opencode_go_healthy,
        default_model="opencode-go:glm-5.3",
        description="Execute an agentic task via OpenCode Go API. Code-focused agent for local development.",
        accepts_extra=True, oai_compatible=True,
        param_name="task",
        error_reason="API key not configured",
        model_prefix="opencode-go:*",
    ),
    ProviderConfig(
        name="qwen_cli", display_name="Qwen", category="cli",
        detector=is_qwen_model,
        call_direct=proxy.call_qwen_direct,
        call_streaming=proxy.call_qwen_streaming,
        health_fn=_is_qwen_healthy,
        default_model="qwen:coder-model",
        description="Qwen Code CLI — free but unreliable. Good for low-stakes tasks.",
        accepts_extra=True, oai_compatible=True,
        param_name="task",
        error_reason="auth not ready",
        model_prefix="qwen:*",
    ),
    # Claude is the default/fallback — detector=None, must be LAST
    ProviderConfig(
        name="claude", display_name="Claude", category="chat",
        detector=None,
        call_direct=proxy.call_api_direct,
        call_streaming=proxy.call_api_streaming,
        health_fn=_is_claude_healthy,
        default_model=proxy.DEFAULT_MODEL,
        description="Claude (Anthropic) — paid with heavy rate limits. Use when deep thinking or complex reasoning is necessary.",
        error_reason="auth not ready",
        model_prefix=None,
    ),
]


def detect_provider(model: str) -> ProviderConfig:
    """Find the provider for a model name. Falls back to Claude (last entry, detector=None)."""
    for p in REGISTRY:
        if p.detector is not None and p.detector(model):
            return p
    fallbacks = [p for p in REGISTRY if p.detector is None]
    if not fallbacks:
        raise RuntimeError("No fallback provider (detector=None) configured in registry")
    return fallbacks[0]


def _wrap_health_fns() -> None:
    """AND each provider's health_fn with its configured `enabled` flag.

    Idempotent: safe to call more than once (re-imports, test reloads).
    `get_provider_setting` is imported lazily to avoid an import cycle
    (provider_config imports REGISTRY for known names).
    """
    from proxy_internal import provider_config

    for p in REGISTRY:
        if getattr(p.health_fn, "_enabled_wrapped", False):
            continue
        original = p.health_fn
        name = p.name

        def _wrapped(_original=original, _name=name):
            if not provider_config.get_provider_setting(_name, "enabled", True):
                return False
            return _original()

        _wrapped._enabled_wrapped = True
        _wrapped.__name__ = f"_enabled_gated_{name}"  # aids tracebacks
        p.health_fn = _wrapped


_wrap_health_fns()
