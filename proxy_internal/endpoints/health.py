"""Health check endpoint."""
import os

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    import proxy  # lazy
    from proxy_internal import concurrency
    from proxy_internal import provider_config
    from proxy_internal.registry import REGISTRY

    # REGISTRY-driven view: one entry per provider, respecting enabled flag.
    providers_block = {}
    for p in REGISTRY:
        enabled = provider_config.get_provider_setting(p.name, "enabled", True)
        providers_block[p.name] = {
            "healthy": p.health_fn(),   # already factors in enabled
            "enabled": enabled,
            "category": p.category,
            "default_model": p.default_model,
        }

    any_healthy = any(v["healthy"] for v in providers_block.values())

    # Legacy keys preserved — existing callers and the dashboard read these.
    def _resolve_command_api_configured() -> bool:
        resolver = getattr(proxy, "_resolve_command_code_api_key", None)
        if callable(resolver):
            try:
                return resolver() is not None
            except Exception:
                return False
        return getattr(proxy, "command_code_api_key", None) is not None

    def _resolve_opencode_zen_configured() -> bool:
        try:
            _m, _u, key, _p = proxy._resolve_opencode("opencode:probe")
            return key is not None
        except Exception:
            return proxy.opencode_api_key is not None

    def _resolve_opencode_go_configured() -> bool:
        try:
            _m, _u, key, _p = proxy._resolve_opencode("opencode-go:glm-5.3")
            return key is not None
        except Exception:
            return proxy.opencode_go_api_key is not None
    accounts_info = proxy.antigravity_auth.get_all_accounts_info()
    try:
        antigravity_ready = proxy.antigravity_auth.is_ready or proxy.ensure_antigravity_discovery()
    except Exception:
        antigravity_ready = proxy.antigravity_auth.is_ready
    try:
        codex_ready = proxy.codex_auth.is_ready or proxy.ensure_codex_discovery()
    except Exception:
        codex_ready = proxy.codex_auth.is_ready
    try:
        claude_ready = proxy.is_claude_auth_ready()
    except Exception:
        claude_ready = proxy.auth.is_ready
    redacted_accounts = [
        {
            "is_ready": bool(acc.get("is_ready")),
            "error_count": int(acc.get("error_count", 0)),
        }
        for acc in accounts_info
        if isinstance(acc, dict)
    ]
    return {
        "status": "healthy" if any_healthy else "warming_up",
        "providers": providers_block,
        # Per-provider concurrency pools: limit / active / queued plus running
        # totals. Only providers with a configured max_concurrent appear.
        "concurrency": concurrency.snapshot(),
        "claude": {
            "path": os.path.basename(proxy.CLAUDE_PATH) if proxy.CLAUDE_PATH else None,
            "auth_cached": claude_ready,
            "auth_source": proxy.claude_auth_source(),
        },
        "codex": {
            "path": os.path.basename(proxy.CODEX_PATH) if proxy.CODEX_PATH else None,
            "auth_cached": codex_ready,
            "auth_source": proxy.codex_auth_source(),
            "token_expired": proxy.codex_auth.is_expired() if proxy.codex_auth.is_ready else None,
        },
        "antigravity": {
            "auth_cached": antigravity_ready,
            "auth_source": proxy.antigravity_auth_source(),
            "email": None,
            "project_id": None,
            "token_expired": proxy.antigravity_auth.is_expired() if proxy.antigravity_auth.is_ready else None,
            "accounts": redacted_accounts,
            "account_count": len(accounts_info),
        },
        "openrouter_configured": proxy._resolve_openrouter_api_key() is not None,
        "openrouter_source": proxy.openrouter_key_source(),
        "vertex_configured": proxy._resolve_vertex_api_key() is not None,
        "vertex_source": proxy.vertex_key_source(),
        "kilo_configured": proxy.kilo_api_key is not None,
        "ollama_configured": proxy._resolve_ollama_api_key() is not None,
        "ollama_source": proxy.ollama_key_source(),
        "lmstudio_configured": True,  # LM Studio is always local
        "zai_configured": proxy.zai_api_key is not None,
        "xiaomi_configured": proxy.xiaomi_api_key is not None,
        "opencode_zen_configured": _resolve_opencode_zen_configured(),
        "opencode_go_configured": _resolve_opencode_go_configured(),
        "fireworks_configured": proxy.fireworks_api_key is not None,
        "nvidia_configured": proxy.nvidia_api_key is not None,
        "bytedance_configured": proxy.bytedance_api_key is not None,
        "command_api_configured": _resolve_command_api_configured(),
        "qwen": {
            "auth_cached": proxy.qwen_auth.is_ready,
            "token_expired": proxy.qwen_auth.is_expired() if proxy.qwen_auth.is_ready else None,
            "has_refresh_token": proxy.qwen_auth.refresh_token is not None,
        },
    }
