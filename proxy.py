"""
OpenAI-compatible proxy using Claude Max subscription, OpenAI Codex CLI subscription, and/or Kilo Gateway API.

Architecture:
  Startup: spawns ONE claude --print OR codex from clean temp dir → captures auth
  Per-request: direct aiohttp call to api.anthropic.com, api.openai.com, or api.kilo.ai via persistent session

Optimizations:
  - Clean temp dir capture: ~350 chars system-reminder vs ~16K (no CLAUDE.md bloat)
  - Persistent aiohttp session: reuses TCP+TLS connection (saves ~200-500ms/request)
  - Direct API calls: no subprocess per request

⚠️  TOS WARNING: Codex auth capture intercepts OAuth tokens from the Codex CLI.
    This may violate OpenAI's Terms of Service. Use at your own risk.

⚠️  TOS WARNING: Antigravity auth uses Google OAuth to access Google's Antigravity IDE API.
    This may violate Google's Terms of Service. Use at your own risk.
"""

import asyncio
import inspect
import contextvars
import gc
import html
import json
import random
import secrets
import time
import uuid
import logging
import shutil
import os
import threading
import sys

# When run as `python proxy.py`, Python registers this module as '__main__' but
# NOT as 'proxy'. Internal sub-modules (e.g. config_runtime endpoints) do
# `import proxy` at call time, which would cause Python to import proxy.py a
# *second* time as a separate module object - giving each endpoint its own copy
# of the module-level globals (timeout_cutoff_seconds, etc.) that never reflects
# runtime changes made to the __main__ copy. Registering early under both names
# prevents the stale duplicate and ensures all `import proxy` calls return the
# live, running module.
sys.modules.setdefault("proxy", sys.modules[__name__])
import statistics
from collections import deque
from contextlib import asynccontextmanager, suppress
from typing import Optional
from datetime import datetime, timedelta

# Raise gen0 threshold to reduce stop-the-world pause frequency.
# Short-lived request objects die before gen0 reaches 10k, so this
# reduces pauses without accumulating long-lived garbage.
gc.set_threshold(10000, 20, 20)

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from aiohttp import web
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy")

ADMIN_CSRF_COOKIE = "proxy_csrf"
ADMIN_CSRF_TOKEN = secrets.token_urlsafe(32)

# --- Phase 1 refactor: re-exports from proxy_internal package ---
# These functions and constants used to live in this file; they were moved
# to proxy_internal/ to start breaking up the monolith.  Every symbol below
# is re-exported at module scope so that (a) tests that call
# proxy.<name>(...) keep working, (b) mcp_server.py and any downstream code
# that does ``from proxy import <name>`` keeps resolving, and (c) internal
# call sites in the rest of this file keep working without edits.
from proxy_internal._json import json_dumps
from proxy_internal.config_store import (
    _load_config,
    _load_max_retries,
    _save_config,
)
from proxy_internal.sse_utils import (
    _close_async_iterator,
    _format_anthropic_sse,
    make_sse_error_chunk,
    preflight_sse_stream,
    yield_sse_error,
)
from proxy_internal.output_sanitize import sanitize_response, sanitize_sse_stream
from proxy_internal import concurrency
from proxy_internal.sse_parser import (
    IncrementalSSEDecoder,
    iter_sse_frames,
    parse_sse_frame,
)
from proxy_internal.message_normalize import (
    _normalize_chat_messages,
    build_oai_messages,
    _sanitize_claude_template as _sanitize_claude_template_impl,
)

_sanitize_claude_template = _sanitize_claude_template_impl

from proxy_internal.model_detect import (
    parse_model_list,
    is_ollama_model,
    is_lmstudio_model,
    is_openrouter_model,
    is_codex_model,
    is_antigravity_model,
    is_zai_model,
    is_xiaomi_model,
    is_opencode_model,
    is_opencode_go_model,
    is_qwen_model,
    is_fireworks_model,
    is_nvidia_model,
    is_kilo_model,
    is_bytedance_model,
    is_command_code_model,
    is_command_code_api_model,
    is_vertex_model,
    normalize_model_name,
)

# --- Phase 2 facade imports: retry helpers + streaming ---
from proxy_internal.streaming import (
    passthrough_sse,
    passthrough_sse_with_rewrite,
)

_passthrough_sse_original = passthrough_sse
from proxy_internal.message_normalize import _model_uses_oai_messages

# --- Phase 3 facade imports: provider modules ---
from proxy_internal.providers.nvidia import (
    call_nvidia_direct,
    call_nvidia_streaming,
    _resolve_nvidia_model,
    _NVIDIA_MODEL_ALIASES,
)
from proxy_internal.providers.openrouter import (
    call_openrouter_direct,
    call_openrouter_streaming,
)
from proxy_internal.providers.bytedance import (
    call_bytedance_direct,
    call_bytedance_streaming,
)
from proxy_internal.providers.command_code import (
    call_command_code_direct,
    call_command_code_streaming,
    call_command_code_api_direct,
    call_command_code_api_streaming,
    _fetch_cmd_models,
    _is_cmd_authenticated,
)
from proxy_internal.providers.kilo import (
    call_kilo_direct,
    call_kilo_streaming,
    _resolve_kilo_model,
    _fetch_kilo_models,
)
from proxy_internal.providers.vertex import (
    call_vertex_direct,
    call_vertex_streaming,
    close_vertex_clients,
)
from proxy_internal.providers.ollama import (
    call_ollama_direct,
    call_ollama_streaming,
)
from proxy_internal.providers.lmstudio import (
    call_lmstudio_direct,
    call_lmstudio_streaming,
)
from proxy_internal.providers.zai import (
    call_zai_direct,
    call_zai_streaming,
)
from proxy_internal.providers.xiaomi import call_xiaomi_direct, call_xiaomi_streaming
from proxy_internal.providers.opencode import (
    call_opencode_direct,
    call_opencode_streaming,
    _resolve_opencode,
)
from proxy_internal.providers.qwen import call_qwen_direct, call_qwen_streaming
from proxy_internal.providers.fireworks import (
    call_fireworks_direct,
    call_fireworks_streaming,
    _resolve_fireworks_model,
)
from proxy_internal.providers.codex import (
    call_codex_direct,
    call_codex_messages_passthrough,
    call_codex_streaming,
    _cleanup_isolated_home,
    _create_isolated_codex_home,
    _get_codex_command,
)
from proxy_internal.providers.claude import (
    call_api_direct,
    call_api_streaming,
    call_claude_messages_passthrough,
    _build_api_body,
)
from proxy_internal.providers.antigravity import (
    call_antigravity_direct,
    call_antigravity_streaming,
    _get_antigravity_model_id,
)
from proxy_internal.providers.antigravity_structured import (
    call_antigravity_structured,
    stream_antigravity_structured,
)


# --- Phase 3.5 facade imports: retry helpers ---
from proxy_internal.retry import (
    _open_stream_with_retry,
    _with_retry as _retry_impl,
)


async def _with_retry(func, *args, operation: str = "", request_id: str = "", **kwargs):
    """Compatibility wrapper for retry logic.
    Provider code calls ``await proxy._with_retry(_do_call, operation=..., request_id=..., **retry_kwargs)``.
    The shared implementation ``proxy_internal.retry._with_retry`` expects the callable
    and the keyword arguments ``operation`` and ``request_id`` (plus any retry settings).
    Positional ``*args`` are ignored because the original retry helper did not forward them.
    """
    # Forward to the shared implementation.
    return await _retry_impl(
        func,
        operation=operation,
        request_id=request_id,
        **kwargs,
    )


# --- Phase 4 facade imports: auth cache classes + loaders ---
# --- Phase 4 facade imports: auth cache classes + loaders ---
from proxy_internal.auth.base import AuthCache
from proxy_internal.auth.codex import CodexAuthCache
from proxy_internal.auth.qwen import QwenAuthCache, load_qwen_auth
from proxy_internal.auth.antigravity import (
    AntigravityAccount,
    AntigravityAuthCache,
    get_antigravity_headers,
    load_antigravity_auth,
    verify_antigravity_client_pair,
    _save_antigravity_auth,
)

# --- Phase 5 facade imports: FastAPI schemas + routers ---
from proxy_internal.schemas import ChatRequest

# --- Phase 6 facade imports: interceptors + tailscale ---
from proxy_internal.interceptors import (
    start_interceptor,
    _claude_auth_refresh_loop,
    start_codex_interceptor,
    load_opencode_key,
    recapture_claude_auth,
)
from proxy_internal.tailscale import _setup_tailscale_serve, get_tailscale_fqdn

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_CODEX_MODEL = "gpt-6-sol"
CODEX_FAST_REASONING_EFFORT = "low"
INTERCEPTOR_PORT = 9999
MCP_INTERCEPTOR_PORT = 9997  # Different port for MCP mode to avoid conflicts
_interceptor_port: int = INTERCEPTOR_PORT  # Actual port used (set by start_interceptor)

CLAUDE_PATH = shutil.which("claude")
if CLAUDE_PATH:
    logger.info(f"Using Claude CLI: {CLAUDE_PATH}")
else:
    logger.warning("Claude CLI not found on PATH - Claude provider unavailable")

CODEX_PATH = shutil.which("codex")
if CODEX_PATH:
    logger.info(f"Using Codex CLI: {CODEX_PATH}")
else:
    logger.warning("Codex CLI not found on PATH - Codex provider unavailable")

COMMAND_CODE_PATH = shutil.which("cmd")
if COMMAND_CODE_PATH:
    logger.info(f"Using Command Code CLI: {COMMAND_CODE_PATH}")

# --- OpenCode (Zen API at opencode.ai/zen/v1, Go API at opencode.ai/zen/go/v1) ---
OPENCODE_AUTH_FILE = os.path.expanduser("~/.local/share/opencode/auth.json")
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1"
OPENCODE_GO_URL = "https://opencode.ai/zen/go/v1"

# --- NVIDIA NIM (free tier at integrate.api.nvidia.com) ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# --- Qwen Code CLI (OAuth via portal.qwen.ai) ---
QWEN_CREDS_FILE = os.path.expanduser("~/.qwen/oauth_creds.json")
QWEN_BASE_URL = "https://portal.qwen.ai/v1"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"


def create_session() -> aiohttp.ClientSession:
    """Create an aiohttp session with connection pooling for low-latency reuse."""
    connector = aiohttp.TCPConnector(
        limit=20, enable_cleanup_closed=True, ttl_dns_cache=300, keepalive_timeout=30
    )
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        json_serialize=_json_serialize,
    )


def _json_serialize(obj: object) -> str:
    """Serialize JSON payloads using the fastest available backend."""
    return json_dumps(obj).decode("utf-8")


# --- Shared streaming / message utilities ---

_CHAT_ALLOWED_EXTRA = {
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "logit_bias",
    "n",
    "reasoning_effort",
    "reasoning",
    "thinking",
}


# --- Token encryption ---


# --- Retry policy ---
#
# One uniform retry policy, tunable at runtime via the dashboard.
# ``max_retries`` is the number of retry attempts AFTER the first try, so the
# total call count is ``max_retries + 1``. Default 1 (one try, one retry on
# failure).
#
# Scope of ``_with_retry``: wrap the network POST + status-validation step of
# every OpenAI-compatible upstream (OpenRouter, Ollama, Z.AI, Xiaomi,
# OpenCode, Qwen, Fireworks, NVIDIA NIM). For streaming providers we only
# retry the pre-first-byte phase - once upstream bytes are flowing we cannot
# cleanly resume, and a mid-stream error is surfaced as an SSE error chunk via
# the existing ``yield_sse_error`` helper.
#
# NOT wrapped (intentional exclusions):
#   - ``call_api_*`` (Claude native) - MITM auth layer handles its own
#     recapture/retries on auth failures.
#   - ``call_codex_*`` - spawns the Codex CLI subprocess; retry semantics are
#     owned by the CLI.
#   - ``call_antigravity_*`` - already has multi-account OAuth fallback.
#
# Retried on:
#   - network exceptions (``aiohttp.ClientConnectorError``,
#     ``aiohttp.ServerDisconnectedError``, ``asyncio.TimeoutError``,
#     ``ConnectionError`` and its subclasses - ``ConnectionResetError``,
#     ``ConnectionAbortedError``, ``BrokenPipeError``)
#   - ``HTTPException`` whose ``status_code`` is in ``_DEFAULT_RETRY_STATUSES``
#     (429, 500, 502, 503, 504)
#
# NOT retried on:
#   - 401/403 (auth broken - fail fast)
#   - other 4xx
#   - bare ``OSError`` (e.g. ``PermissionError``, ``FileNotFoundError``,
#     ``IsADirectoryError``) - these are fatal configuration problems, never
#     transient; retrying them only masks the root cause
#   - any other exception type

# --- Auth cache + persistent session ---

auth = AuthCache()

# Persistent session dedicated to Ollama (Cloud + local) - avoids per-request TLS handshakes.
ollama_session: Optional[aiohttp.ClientSession] = None

# Persistent session for LM Studio (local only).
lmstudio_session: Optional[aiohttp.ClientSession] = None

# Shared session for OpenAI-compatible third-party providers.
third_party_session: Optional[aiohttp.ClientSession] = None

# Cached template parts - extracted once when auth is captured, reused on every request.
# Avoids copy.deepcopy(auth.body_template) overhead per inference call.
_cached_billing_block: Optional[dict] = None
_cached_auth_blocks: list = []

# --- Codex OAuth Auth cache ---

codex_auth = CodexAuthCache()


qwen_auth = QwenAuthCache()

# --- Antigravity (Google IDE) OAuth Auth cache ---

# Antigravity OAuth client identity resolves lazily via
# proxy_internal.cli_client_identity (env first, then CLI-on-disk).
# No hardcoded literals here.
ANTIGRAVITY_OAUTH_ENV_VARS = ("ANTIGRAVITY_CLIENT_ID", "ANTIGRAVITY_CLIENT_SECRET")
_ANTIGRAVITY_OAUTH_CLIENT_CACHE: "Optional[tuple[str, str]]" = None
_ANTIGRAVITY_OAUTH_CLIENT_CACHED = False
_ANTIGRAVITY_OAUTH_SOURCE_LOGGED = False


def get_antigravity_oauth_client() -> "Optional[tuple[str, str]]":
    """Return Antigravity (client_id, client_secret) or None. Lazy, resolved once, no values logged.

    The negative result is cached too: the CLI-on-disk lookup scans a large
    native binary, and re-running it on every auth refresh would repeat the
    same miss. Install the CLI (or set the env vars) and restart to pick up a
    new source.

    When the CLI binary offers several possible id/secret pairings, they are
    settled against the token endpoint (one request per candidate, only in
    that ambiguous case).
    """
    global _ANTIGRAVITY_OAUTH_CLIENT_CACHE, _ANTIGRAVITY_OAUTH_CLIENT_CACHED, _ANTIGRAVITY_OAUTH_SOURCE_LOGGED
    if _ANTIGRAVITY_OAUTH_CLIENT_CACHED:
        return _ANTIGRAVITY_OAUTH_CLIENT_CACHE
    from proxy_internal.cli_client_identity import resolve_antigravity_client, client_source
    pair = resolve_antigravity_client(
        verifier=lambda candidate: verify_antigravity_client_pair(
            candidate, ANTIGRAVITY_REDIRECT_URI
        )
    )
    _ANTIGRAVITY_OAUTH_CLIENT_CACHE = pair
    _ANTIGRAVITY_OAUTH_CLIENT_CACHED = True
    if not _ANTIGRAVITY_OAUTH_SOURCE_LOGGED:
        _ANTIGRAVITY_OAUTH_SOURCE_LOGGED = True
        if pair is not None:
            try:
                source = client_source("antigravity")
            except Exception:
                source = "unknown"
            logger.info(f"Antigravity OAuth client identity resolved via {source}")
        else:
            logger.warning(
                "Antigravity OAuth client identity unresolved; set "
                "ANTIGRAVITY_CLIENT_ID/ANTIGRAVITY_CLIENT_SECRET"
            )
    return pair


def antigravity_oauth_unresolved_error() -> str:
    return (
        "Antigravity OAuth client identity unresolved. Set ANTIGRAVITY_CLIENT_ID and "
        "ANTIGRAVITY_CLIENT_SECRET env vars, or install the agy CLI so "
        "on-disk extraction can succeed."
    )


def __getattr__(name: str):
    """Legacy literal-name shim: resolve lazily, fail closed with guidance."""
    legacy = {
        "ANTIGRAVITY_CLIENT_ID": ("antigravity", 0),
        "ANTIGRAVITY_CLIENT_SECRET": ("antigravity", 1),
    }
    if name in legacy:
        provider, idx = legacy[name]
        pair = get_antigravity_oauth_client()
        if pair is None:
            raise AttributeError(
                antigravity_oauth_unresolved_error()
            )
        return pair[idx]
    raise AttributeError(f"module 'proxy' has no attribute {name!r}")
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
# Wire parity with the official Antigravity client (pi-provider-antigravity
# DEFAULT_ENDPOINT): the client talks to daily-cloudcode-pa (no sandbox
# subdomain), which is a different host from both PROD and _DAILY above.
ANTIGRAVITY_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com"
ANTIGRAVITY_DEFAULT_PROJECT_ID = "aicode-consumers"
ANTIGRAVITY_VERSION = "1.18.3"
ANTIGRAVITY_OAUTH_PORT = 51121


antigravity_auth = AntigravityAuthCache()

# --- OpenRouter + Round-Robin state ---
_cfg = _load_config()
openrouter_api_key: Optional[str] = _cfg.get("openrouter_api_key") or None
if openrouter_api_key:
    logger.info("OpenRouter API key loaded from config.json")
kilo_api_key: Optional[str] = _cfg.get("kilo_api_key") or None
if kilo_api_key:
    logger.info("Kilo Gateway API key loaded from config.json")
ollama_api_key: Optional[str] = _cfg.get("ollama_api_key") or None
if ollama_api_key:
    logger.info("Ollama Cloud API key loaded from config.json")
lmstudio_port: int = _cfg.get("lmstudio_port", 1234)
zai_api_key: Optional[str] = _cfg.get("zai_api_key") or None
if zai_api_key:
    logger.info("Z.AI API key loaded from config.json")
xiaomi_api_key: Optional[str] = _cfg.get("xiaomi_api_key") or None
if xiaomi_api_key:
    logger.info("Xiaomi API key loaded from config.json")
opencode_api_key: Optional[str] = _cfg.get("opencode_api_key") or None
if opencode_api_key:
    logger.info("OpenCode Zen API key loaded from config.json")
opencode_go_api_key: Optional[str] = _cfg.get("opencode_go_api_key") or None
if opencode_go_api_key:
    logger.info("OpenCode Go API key loaded from config.json")
fireworks_api_key: Optional[str] = _cfg.get("fireworks_api_key") or None
if fireworks_api_key:
    logger.info("Fireworks API key loaded from config.json")
nvidia_api_key: Optional[str] = _cfg.get("nvidia_api_key") or None
if nvidia_api_key:
    logger.info("NVIDIA NIM API key loaded from config.json")
bytedance_api_key: Optional[str] = _cfg.get("bytedance_api_key") or None
if bytedance_api_key:
    logger.info("ByteDance API key loaded from config.json")
vertex_api_key: Optional[str] = _cfg.get("vertex_api_key") or None
if vertex_api_key:
    logger.info("Vertex AI API key loaded from config.json")


def _resolve_command_code_api_key() -> Optional[str]:
    """Resolve Command Code API key: env wins, else config, else discovery.

    Priority: COMMAND_CODE_API_KEY env > encrypted
    ``_cfg['command_code_api_key']`` > ``discover_command_code_key()``
    (Pi same-provider synthetic OAuth/API key, then Pi settings, then CLI
    key). Resolved dynamically per call so late logins are picked up
    without restart. Never logs value.
    """
    env_key = os.environ.get("COMMAND_CODE_API_KEY")
    if env_key:
        return env_key
    cfg_key = _cfg.get("command_code_api_key") or None
    if cfg_key:
        return cfg_key
    from proxy_internal.credential_discovery import discover_command_code_key
    return discover_command_code_key()


command_code_api_key: Optional[str] = _resolve_command_code_api_key()
if command_code_api_key:
    logger.info("Command Code API key loaded")


# --- On-disk credential auto-discovery (Pi agent store, last resort) ---
#
# Priority per credential: explicit env var > encrypted ``_cfg`` value >
# on-disk discovery (``proxy_internal.credential_discovery``). Every resolver
# below runs dynamically per call so a credential that appears on disk later
# is picked up without a restart. ``provider isolation`` is inherited from
# the discovery helpers (each reads only its own provider id). Expired OAuth
# records yield None and count as not-ready; OAuth records are never
# converted into API keys or vice versa. Values are never logged; only
# provider, source class (env/config/disk), and redacted status.
_DISK_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "vertex": "VERTEX_API_KEY",
}
_DISK_KEY_CFG = {
    "openrouter": "openrouter_api_key",
    "ollama": "ollama_api_key",
    "vertex": "vertex_api_key",
}
_DISK_KEY_GLOBAL = {
    "openrouter": "openrouter_api_key",
    "ollama": "ollama_api_key",
    "vertex": "vertex_api_key",
}
_disk_source_logged: dict = {}


def _discover_api_key(provider: str) -> Optional[str]:
    """Return the on-disk key for one provider, or None. Never logs values."""
    from proxy_internal import credential_discovery as _cd

    try:
        if provider == "openrouter":
            return _cd.discover_openrouter_key()
        if provider == "ollama":
            return _cd.discover_ollama_key()
        if provider == "vertex":
            return _cd.discover_vertex_key()
    except Exception:
        return None
    return None


def _api_key_source(provider: str) -> Optional[str]:
    """Return which source wins for a provider: env/config/disk, else None.

    Config means an explicit ``_cfg`` value (or an in-memory global set from
    config); a global populated from disk still reports "disk" because the
    disk record is re-checked first. Never exposes values.
    """
    if os.environ.get(_DISK_KEY_ENV[provider]):
        return "env"
    if _cfg.get(_DISK_KEY_CFG[provider]):
        return "config"
    if _discover_api_key(provider) is not None:
        return "disk"
    global_name = _DISK_KEY_GLOBAL.get(provider)
    if global_name and globals().get(global_name):
        return "config"
    return None


def _sync_discovered_api_keys() -> None:
    """Refresh key globals from env/config/disk priority (no restart needed).

    Providers under ``proxy_internal/providers/*`` read ``proxy.<name>_api_key``
    directly, so the winning value is mirrored onto the global: explicit env
    or ``_cfg`` wins, else the current on-disk record (or None when the
    record is missing/expired/malformed, which also clears a stale
    disk-sourced global). Never logs values.
    """
    for provider, global_name in _DISK_KEY_GLOBAL.items():
        env_key = os.environ.get(_DISK_KEY_ENV[provider]) or None
        cfg_key = _cfg.get(_DISK_KEY_CFG[provider]) or None
        if env_key or cfg_key:
            new_value = env_key or cfg_key
            source = "env" if env_key else "config"
        else:
            new_value = _discover_api_key(provider)
            source = "disk" if new_value is not None else None
        if globals().get(global_name) != new_value:
            globals()[global_name] = new_value
        if source and _disk_source_logged.get(provider) != source:
            _disk_source_logged[provider] = source
            if source == "disk":
                logger.info(f"{provider} credential auto-recognized via disk (redacted)")


def _resolve_openrouter_api_key() -> Optional[str]:
    """Resolve OpenRouter key: env > config > on-disk discovery, per call."""
    _sync_discovered_api_keys()
    return openrouter_api_key


def _resolve_ollama_api_key() -> Optional[str]:
    """Resolve Ollama Cloud key: env > config > on-disk discovery, per call."""
    _sync_discovered_api_keys()
    return ollama_api_key


def _resolve_vertex_api_key() -> Optional[str]:
    """Resolve Vertex key: env > config > on-disk discovery, per call."""
    _sync_discovered_api_keys()
    return vertex_api_key


def openrouter_key_source() -> Optional[str]:
    return _api_key_source("openrouter")


def ollama_key_source() -> Optional[str]:
    return _api_key_source("ollama")


def vertex_key_source() -> Optional[str]:
    return _api_key_source("vertex")


def is_claude_auth_ready() -> bool:
    """Discovery-aware Claude readiness: MITM headers, else fresh Pi OAuth."""
    if auth.is_ready:
        return True
    try:
        from proxy_internal.credential_discovery import discover_claude_oauth
        return discover_claude_oauth() is not None
    except Exception:
        return False


def claude_auth_source() -> Optional[str]:
    """Which source satisfies Claude: config (MITM headers) / disk / None."""
    if auth.is_ready:
        return "config"
    try:
        from proxy_internal.credential_discovery import discover_claude_oauth
        return "disk" if discover_claude_oauth() is not None else None
    except Exception:
        return None


def ensure_codex_discovery() -> bool:
    """Populate Codex OAuth from the Pi store when nothing else is ready."""
    if codex_auth.is_ready:
        return True
    try:
        from proxy_internal.credential_discovery import discover_codex_oauth
        record = discover_codex_oauth()
    except Exception:
        return False
    if not record:
        return False
    codex_auth.apply_discovered_oauth(
        record["access_token"], account_id=record.get("account_id")
    )
    if _disk_source_logged.get("codex") != "disk":
        _disk_source_logged["codex"] = "disk"
        logger.info("Codex OAuth token auto-recognized via disk (redacted)")
    return codex_auth.is_ready


def codex_auth_source() -> Optional[str]:
    if codex_auth.is_ready and _disk_source_logged.get("codex") != "disk":
        return "config"
    if ensure_codex_discovery():
        return "disk" if _disk_source_logged.get("codex") == "disk" else "config"
    return None


_DISK_ANTIGRAVITY_EMAIL = "pi-autodiscovery"


def ensure_antigravity_discovery() -> bool:
    """Recapture Antigravity auth without restart: own file first, then matching discovery.

    Dynamic per call so a login appearing later is picked up. Only the
    matching ``antigravity`` provider record is adopted (never another
    provider's); expired records stay not-ready. Never logs values.
    """
    if antigravity_auth.native_keychain:
        return antigravity_auth.is_ready or antigravity_auth.reload_native()
    if antigravity_auth.is_ready:
        return True
    try:
        from proxy_internal.auth.antigravity import resync_antigravity_from_disk
        if resync_antigravity_from_disk():
            if _disk_source_logged.get("antigravity") != "file":
                _disk_source_logged["antigravity"] = "file"
                logger.info("Antigravity auth recaptured via file (redacted)")
            return True
    except Exception:
        pass
    try:
        from proxy_internal.auth.antigravity import _adopt_discovered_antigravity_token
        adopted = _adopt_discovered_antigravity_token()
    except Exception:
        return False
    if not adopted:
        return False
    if _disk_source_logged.get("antigravity") != "disk":
        _disk_source_logged["antigravity"] = "disk"
        logger.info("Antigravity OAuth token auto-recognized via disk (redacted)")
    return antigravity_auth.is_ready


def antigravity_auth_source() -> Optional[str]:
    if antigravity_auth.native_keychain:
        return "keychain" if ensure_antigravity_discovery() else None
    if antigravity_auth.is_ready and _disk_source_logged.get("antigravity") not in ("disk", "file"):
        return "config"
    if ensure_antigravity_discovery():
        return _disk_source_logged.get("antigravity") if _disk_source_logged.get("antigravity") in ("disk", "file") else "config"
    return None


def _log_auth_source(provider_label: str, source: Optional[str], interactive_hint: str) -> None:
    """Log which auth source won: file / discovery / interactive-needed. Never logs values."""
    if source == "keychain":
        logger.info(f"{provider_label} auth ready via agy Keychain session (read-only)")
    elif source == "config":
        logger.info(f"{provider_label} auth ready via file (redacted)")
    elif source == "file":
        logger.info(f"{provider_label} auth ready via file (redacted)")
    elif source == "disk":
        logger.info(f"{provider_label} auth ready via discovery (redacted)")
    else:
        logger.info(f"{provider_label} auth not configured -- {interactive_hint}")


timeout_routing_enabled: bool = _cfg.get("timeout_routing_enabled", True)
timeout_cutoff_seconds: float = _cfg.get("timeout_cutoff_seconds", 6.0)
max_total_seconds: float = _cfg.get("max_total_seconds", 9.0)
max_retries: int = _load_max_retries(_cfg)
logger.info(
    f"Timeout routing: {'enabled' if timeout_routing_enabled else 'disabled'} (TTFT cutoff {timeout_cutoff_seconds}s, max total {max_total_seconds}s)"
)
logger.info(f"Retry policy: max_retries={max_retries}")

# --- Reasoning override (dashboard toggle) ---
#
# When CCS (Claude Code Switcher) routes through this proxy on the ``proxy``
# profile, its per-tier ``thinking.tier_defaults`` only flow to CCS's own
# cliproxy providers - requests into us arrive with ``thinking`` absent, and
# the request pipeline then forces ``thinking={"type":"disabled"}``. The
# override below lets dashboard operators flip the default back on at a
# configurable effort level so Claude Code gets reasoning again even through
# CCS. OFF by default so existing callers see unchanged behavior.
reasoning_override_enabled: bool = bool(_cfg.get("reasoning_override_enabled", False))
reasoning_override_level: str = _cfg.get("reasoning_override_level", "medium")
if reasoning_override_level not in ("low", "medium", "high"):
    reasoning_override_level = "medium"
logger.info(
    f"Reasoning override: {'enabled' if reasoning_override_enabled else 'disabled'} "
    f"(level={reasoning_override_level})"
)

# Anthropic ``thinking.budget_tokens`` bucket per override level. Numbers are
# order-of-magnitude estimates used by Claude / Anthropic-shaped bodies; OAI
# providers ignore the field entirely.
_THINKING_BUDGET_BY_LEVEL = {"low": 2048, "medium": 8192, "high": 24576}


def _override_thinking_payload(max_tokens: int = 0) -> dict:
    """Return the Anthropic-shaped ``thinking`` dict to inject when override is on."""
    budget = _THINKING_BUDGET_BY_LEVEL.get(reasoning_override_level, 8192)
    # Clamp budget to max_tokens-1 when max_tokens is set and triggers clamping.
    # Anthropic requires budget_tokens >= 1024 AND budget_tokens < max_tokens.
    # When max_tokens in (0, 1024] we can't satisfy both, so disable thinking.
    if 0 < max_tokens <= 1024:
        return {"type": "disabled"}
    if max_tokens > 0 and max_tokens <= budget:
        budget = max_tokens - 1
    return {"type": "enabled", "budget_tokens": budget}


def _active_codex_reasoning_effort() -> str:
    """Return the effective ``model_reasoning_effort`` for Codex requests.

    When the dashboard reasoning override is on, Codex uses the override level
    (``low``/``medium``/``high``) rather than the hard-coded fast-mode default.
    """
    if reasoning_override_enabled:
        return reasoning_override_level
    return CODEX_FAST_REASONING_EFFORT


_round_robin_counter: int = 0
reasoning_rewrite_enabled: bool = _cfg.get("reasoning_rewrite_enabled", False)
logger.info(
    f"Reasoning rewrite: {'enabled' if reasoning_rewrite_enabled else 'disabled'}"
)

# MCP mode flag - set True by the MCP server entry point when this module is
# reused as an MCP tool backend. In MCP mode, reasoning/thinking is left enabled by default, SSE scrubbing is bypassed,
# and the reasoning→NPC-dialogue rewrite is disabled - MCP callers are general-purpose
# agents, not SkyrimNet game NPCs.
#
# Known limitation: OAI-compatible providers (OpenRouter, Z.AI, OpenCode, Qwen, Xiaomi,
# Ollama, NVIDIA) still strip ``thinking``/``reasoning`` from payloads in MCP mode because
# those upstream APIs reject unknown params with 400 errors. Claude (Anthropic native path),
# Antigravity fully preserves reasoning/thought blocks in MCP mode.
MCP_MODE: bool = False

# Port the proxy serves HTTP on. Default matches the launcher INI default
# (Port=8000); overridden from --port in __main__ before serving, and used by
# the OAuth callback pages below so redirects land on the configured port.
PROXY_PORT: int = 8000

# Contextvar for per-request rewrite context (set in streaming path, read by passthrough_sse wrapper)
_rewrite_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "rewrite_ctx", default={}
)


def _passthrough_sse_dispatch(resp, request_id, provider_name, start, **kw):
    """Dispatch to rewrite-aware passthrough when rewrite context is set, otherwise original.
    In MCP mode, always use the original passthrough (no NPC dialogue rewrite)."""
    ctx = _rewrite_ctx.get()
    if (
        not MCP_MODE
        and not reasoning_override_enabled
        and reasoning_rewrite_enabled
        and ctx
    ):
        return passthrough_sse_with_rewrite(
            resp,
            request_id,
            provider_name,
            start,
            system_prompt=ctx.get("system_prompt"),
            model=ctx.get("model", ""),
        )
    return _passthrough_sse_original(resp, request_id, provider_name, start, **kw)


# Replace the module-level attribute so all providers use the dispatch
passthrough_sse = _passthrough_sse_dispatch


# --- Per-model latency / reliability stats (in-memory, rolling window) ---


class ModelStatsTracker:
    """Rolling window of time-to-first-token measurements per model."""

    def __init__(self, window: int = 50):
        self._records: dict = {}  # model -> deque of (ttft_s, success)
        self._window = window

    def record(self, model: str, ttft: float, success: bool):
        if model not in self._records:
            self._records[model] = deque(maxlen=self._window)
        self._records[model].append((ttft, success))

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for on-disk persistence."""
        return {
            "window": self._window,
            "records": {
                model: [[t, s] for t, s in records]
                for model, records in self._records.items()
            },
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore state from a previously-saved dict. Preserves rolling-window
        maxlen - records exceeding the window are silently truncated to the tail."""
        window = int(data.get("window", self._window))
        self._window = window
        self._records = {}
        for model, entries in data.get("records", {}).items():
            dq = deque(maxlen=window)
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    dq.append((float(entry[0]), bool(entry[1])))
            if dq:
                self._records[model] = dq

    def get_stats(self) -> dict:
        out = {}
        for model, records in self._records.items():
            if not records:
                continue
            ttfts = [t for t, s in records if s]
            success_count = sum(1 for _, s in records if s)
            out[model] = {
                "samples": len(records),
                "success_rate": round(success_count / len(records), 3),
                "median_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
                "p90_ttft_s": round(sorted(ttfts)[int(len(ttfts) * 0.9)], 3)
                if len(ttfts) >= 10
                else None,
            }
        return out

    def get_reliable_model(self, exclude) -> Optional[str]:
        """Return a reliable fallback model, excluding the given one(s).

        `exclude` may be a single model name (str) or an iterable of names to
        skip - used by the cascading fallback path in `_with_timeout_routing`,
        which needs to keep asking for "the next best" model after each failure.

        Selection strategy (tiered):
          1. **Fast tier** - models with median TTFT < 3.0s AND success_rate >= 0.8.
             Pick one at random, weighted by score. Load-balances across good models
             so we don't hammer the single fastest one and trip its rate limits.
          2. **Best-effort tier** - if no model meets the fast-tier threshold, fall
             back to the single highest-scoring candidate (original behavior).

        Score = success_rate * 0.6 + speed_score * 0.3 + sample_confidence * 0.1
        Uses both TTFT stats and full request_stats for richer signal.
        """
        # Normalize exclude to a set for uniform membership checks.
        if exclude is None:
            exclude_set = set()
        elif isinstance(exclude, str):
            exclude_set = {exclude}
        else:
            exclude_set = set(exclude)

        candidates = []  # all viable (success_rate >= 0.7)
        fast_candidates = []  # subset: median_ttft < 3.0 AND success_rate >= 0.8
        for model, records in self._records.items():
            if model in exclude_set or len(records) < 5:
                continue
            ttfts = [t for t, s in records if s]
            if not ttfts:
                continue
            # TTFT success rate from this tracker
            ttft_success = sum(1 for _, s in records if s) / len(records)

            # Blend in overall request success rate from request_stats
            req_success, req_median_lat = request_stats.get_model_reliability(model)
            req_samples = request_stats.get_model_sample_count(model)

            # Combined success rate: weight request_stats more if it has data
            if req_samples >= 5:
                success_rate = ttft_success * 0.4 + req_success * 0.6
            else:
                success_rate = ttft_success

            if success_rate < 0.7:
                continue

            median_ttft = statistics.median(ttfts)
            # Speed score: normalize to 0-1 range (0s = 1.0, 10s+ = 0.0)
            speed_score = max(0.0, 1.0 - median_ttft / 10.0)
            # Sample confidence: more samples = more trust (caps at 1.0 at 30 samples)
            total_samples = len(records) + req_samples
            sample_confidence = min(1.0, total_samples / 30.0)

            score = success_rate * 0.6 + speed_score * 0.3 + sample_confidence * 0.1
            candidates.append((model, score))
            if median_ttft < 3.0 and success_rate >= 0.8:
                fast_candidates.append((model, score))

        # Fast tier: weighted random pick among models that consistently beat 3s TTFT.
        if fast_candidates:
            models = [m for m, _ in fast_candidates]
            weights = [s for _, s in fast_candidates]
            return random.choices(models, weights=weights, k=1)[0]

        # Best-effort tier: no fast candidates, pick the single best.
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]


# Note: _stats_lock was removed in Fix 4 (stats persistence race).
# Snapshots are now built on the event loop; only file I/O runs in the thread.

model_stats = ModelStatsTracker()


# --- Full request stats tracker (all providers, streaming + direct) ---


class RequestStatsTracker:
    """Track request counts, latencies, and error rates per model per mode."""

    def __init__(self, window: int = 100):
        self._window = window
        self._data: dict = {}  # model -> mode -> {count, errors, latencies: deque}
        self._global_count = 0
        self._global_errors = 0
        self._start_time = time.time()

    def _ensure(self, model: str, mode: str):
        if model not in self._data:
            self._data[model] = {}
        if mode not in self._data[model]:
            self._data[model][mode] = {
                "count": 0,
                "errors": 0,
                "latencies": deque(maxlen=self._window),
            }

    def record(self, model: str, mode: str, latency: float, success: bool):
        self._ensure(model, mode)
        bucket = self._data[model][mode]
        bucket["count"] += 1
        bucket["latencies"].append(latency)
        self._global_count += 1
        if not success:
            bucket["errors"] += 1
            self._global_errors += 1

    def get_stats(self) -> dict:
        by_model = {}
        for model, modes in self._data.items():
            by_model[model] = {}
            for mode, bucket in modes.items():
                lats = list(bucket["latencies"])
                count = bucket["count"]
                errors = bucket["errors"]
                by_model[model][mode] = {
                    "requests": count,
                    "errors": errors,
                    "error_rate": round(errors / count, 3) if count else 0,
                    "median_latency_s": round(statistics.median(lats), 3)
                    if lats
                    else None,
                    "p90_latency_s": round(sorted(lats)[int(len(lats) * 0.9)], 3)
                    if len(lats) >= 10
                    else None,
                    "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
                }
        uptime = time.time() - self._start_time
        return {
            "global": {
                "total_requests": self._global_count,
                "total_errors": self._global_errors,
                "error_rate": round(self._global_errors / self._global_count, 3)
                if self._global_count
                else 0,
                "uptime_s": round(uptime, 1),
                "requests_per_minute": round(self._global_count / (uptime / 60), 2)
                if uptime > 0
                else 0,
            },
            "by_model": by_model,
        }

    def get_model_reliability(self, model: str) -> tuple[float, float]:
        """Return (success_rate, median_latency) across all modes for a model."""
        total, errors, lats = 0, 0, []
        for mode, bucket in self._data.get(model, {}).items():
            total += bucket["count"]
            errors += bucket["errors"]
            lats.extend(bucket["latencies"])
        success_rate = (total - errors) / total if total else 0
        median_lat = statistics.median(lats) if lats else 999
        return success_rate, median_lat

    def get_model_sample_count(self, model: str) -> int:
        return sum(b["count"] for b in self._data.get(model, {}).values())

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for on-disk persistence."""
        data_out = {}
        for model, modes in self._data.items():
            data_out[model] = {}
            for mode, bucket in modes.items():
                data_out[model][mode] = {
                    "count": bucket["count"],
                    "errors": bucket["errors"],
                    "latencies": list(bucket["latencies"]),
                }
        return {
            "window": self._window,
            "data": data_out,
            "global_count": self._global_count,
            "global_errors": self._global_errors,
            "start_time": self._start_time,
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore state from a previously-saved dict. Preserves rolling-window
        maxlen on latencies; counts and start_time carry over so
        requests_per_minute stays meaningful across restarts."""
        window = int(data.get("window", self._window))
        self._window = window
        self._data = {}
        for model, modes in data.get("data", {}).items():
            self._data[model] = {}
            for mode, bucket in modes.items():
                lats = deque(maxlen=window)
                for lat in bucket.get("latencies", []):
                    try:
                        lats.append(float(lat))
                    except (TypeError, ValueError):
                        pass
                self._data[model][mode] = {
                    "count": int(bucket.get("count", 0)),
                    "errors": int(bucket.get("errors", 0)),
                    "latencies": lats,
                }
        self._global_count = int(data.get("global_count", 0))
        self._global_errors = int(data.get("global_errors", 0))
        # Preserve the original start_time so uptime/rpm stay cumulative across
        # restarts. Fall back to now() if the saved file has no timestamp.
        self._start_time = float(data.get("start_time", time.time()))


request_stats = RequestStatsTracker()


# --- Stats persistence to disk -----------------------------------------------
# Historical stats survive proxy restarts so smart routing doesn't start cold
# every time. File is rewritten atomically via tmp+rename to avoid corruption.

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")
_STATS_FILE_VERSION = 1
_stats_persist_task: Optional[asyncio.Task] = None
_stats_persist_interval_s = 60.0


def _load_stats_from_disk() -> None:
    """Restore ModelStatsTracker and RequestStatsTracker state from STATS_FILE.
    Missing or corrupted files are ignored - we just start with empty stats."""
    if not os.path.exists(STATS_FILE):
        logger.info(f"No stats file at {STATS_FILE} - starting with empty history")
        return
    try:
        with open(STATS_FILE, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            f"Stats file {STATS_FILE} unreadable ({e}) - starting with empty history"
        )
        return
    try:
        if payload.get("version") != _STATS_FILE_VERSION:
            logger.warning(
                f"Stats file version mismatch (got {payload.get('version')}, expected {_STATS_FILE_VERSION}) - ignoring"
            )
            return
        model_stats.load_from_dict(payload.get("model_stats", {}))
        request_stats.load_from_dict(payload.get("request_stats", {}))
        saved_at = payload.get("saved_at")
        age_str = ""
        if isinstance(saved_at, (int, float)):
            age_min = (time.time() - saved_at) / 60.0
            age_str = f", saved {age_min:.1f} min ago"
        logger.info(
            f"Restored stats: {len(model_stats._records)} models in TTFT tracker, "
            f"{request_stats._global_count} total requests{age_str}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to restore stats from {STATS_FILE}: {e} - starting with empty history"
        )


def _snapshot_stats() -> dict:
    """Build stats payload on the event loop (safe snapshot)."""
    return {
        "version": _STATS_FILE_VERSION,
        "saved_at": time.time(),
        "model_stats": model_stats.to_dict(),
        "request_stats": request_stats.to_dict(),
    }


def _write_stats_to_disk(payload: dict) -> None:
    """Atomically write a pre-built stats payload to STATS_FILE (tmp + rename)."""
    tmp_path = STATS_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, STATS_FILE)
    except OSError as e:
        logger.warning(f"Failed to save stats to {STATS_FILE}: {e}")
        # Best-effort cleanup of the tmp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


async def _stats_persist_loop() -> None:
    """Background task: flush stats to disk every _stats_persist_interval_s."""
    while True:
        try:
            await asyncio.sleep(_stats_persist_interval_s)
        except asyncio.CancelledError:
            return
        try:
            # Build snapshot on event loop, file I/O off it.
            payload = _snapshot_stats()
            await asyncio.to_thread(_write_stats_to_disk, payload)
        except Exception as e:
            logger.warning(f"Stats persist loop hiccup: {e}")


# Load historical stats at module import so model_stats and request_stats are
# warm before the first request arrives.
_load_stats_from_disk()


def pick_model_round_robin(models: list[str]) -> str:
    """Pick next model from list using round-robin."""
    global _round_robin_counter
    model = models[_round_robin_counter % len(models)]
    _round_robin_counter += 1
    return model


def _pick_fallback_model(exclude_model=None, *, exclude=None):
    """Select the most reliable model from observed stats, or a safe provider default.

    Accepts either `exclude_model` (legacy, str) or `exclude` (str or iterable).
    Used by the cascading fallback path, which needs to pass an ever-growing
    set of already-tried models. Returns None if no candidate outside the
    excluded set is available - callers should then stop cascading.
    """
    excluded = exclude if exclude is not None else exclude_model
    if excluded is None:
        excluded_set: set = set()
    elif isinstance(excluded, str):
        excluded_set = {excluded}
    else:
        excluded_set = set(excluded)

    best = model_stats.get_reliable_model(exclude=excluded_set)
    if best:
        return best
    # Provider defaults in reliability priority order - only returned if not
    # already tried, so the cascade doesn't loop on the same hard-coded pick.
    defaults = []
    if auth.is_ready:
        defaults.append(DEFAULT_MODEL)
    if antigravity_auth.is_ready:
        defaults.append("antigravity-gemini-2.5-flash")
    if codex_auth.is_ready:
        defaults.append(DEFAULT_CODEX_MODEL)
    for d in defaults:
        if d not in excluded_set:
            return d
    # Legacy behavior: when no exclude set is given and nothing is ready,
    # still return DEFAULT_MODEL so single-shot callers get a non-None result.
    if not excluded_set:
        return DEFAULT_MODEL
    return None


_REASONING_CHUNK_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "thinking")


def _chunk_has_content(chunk) -> bool:
    """Cheap check: does this SSE chunk carry a non-empty visible delta.content?

    Used by timeout routing to detect the first real content chunk (TTFT).
    Role/setup/[DONE] chunks return False; reasoning/thinking-only chunks also
    return False (see ``_chunk_has_reasoning``) so a reasoning-only stream can
    neither satisfy TTFT nor suppress fallback.
    The string-level guard (`startswith`, `'"content"' in chunk`) avoids the
    json.loads cost on the vast majority of chunks that don't have content.
    """
    try:
        if not (
            isinstance(chunk, str)
            and chunk.startswith("data: ")
            and "[DONE]" not in chunk
            and '"content"' in chunk
        ):
            return False
        data = json.loads(chunk[6:])
        delta = data.get("choices", [{}])[0].get("delta", {})
        return bool(delta.get("content"))
    except Exception:
        return False


def _chunk_has_reasoning(chunk) -> bool:
    """Does this SSE chunk carry any reasoning/thinking field?

    Covers ``reasoning``, ``reasoning_content``, ``reasoning_details`` and
    ``thinking``. Kept separate from ``_chunk_has_content`` so the semantics
    are explicit: reasoning-only events are buffered but never count as
    visible content for TTFT/success/fallback decisions.
    """
    try:
        if not (
            isinstance(chunk, str)
            and chunk.startswith("data: ")
            and "[DONE]" not in chunk
        ):
            return False
        if '"reasoning' not in chunk and '"thinking' not in chunk:
            return False
        data = json.loads(chunk[6:])
        delta = data.get("choices", [{}])[0].get("delta", {})
        return any(delta.get(field) for field in _REASONING_CHUNK_FIELDS)
    except Exception:
        return False


def _chunk_is_role_assistant(chunk: str) -> bool:
    """Check whether an SSE chunk sets role="assistant".

    Used by timeout-routing to decide whether synthetic completion chunks
    need an artificial role chunk first (the game's SSE parser expects a
    role-assistant delta before any finish_reason chunk).
    """
    try:
        if not (isinstance(chunk, str) and chunk.startswith("data: ") and '"role"' in chunk):
            return False
        data = json.loads(chunk[6:])
        delta = data.get("choices", [{}])[0].get("delta", {})
        return delta.get("role") == "assistant"
    except Exception:
        return False


def _has_role_chunk(chunks: list) -> bool:
    """Return True if any chunk in the list sets role="assistant"."""
    return any(_chunk_is_role_assistant(c) if isinstance(c, str) else False for c in chunks)


def _synthetic_stop_done(model: str, needs_role: bool = False):
    """Yield SSE chunks for clean completion.

    When *needs_role* is True (no role chunk was sent yet), emit a
    role-assistant chunk before the stop chunk so the game's SSE parser
    doesn't crash on receiving finish_reason as the first event.
    """
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    if needs_role:
        role_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"
    stop_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _is_claude_default_model(model: str) -> bool:
    """True when the given model would default-route to Claude (Anthropic).

    Mirrors the routing decision in ``_chat_completions_inner`` / the
    ``_make_streaming_gen`` else-branch: a model belongs to Claude iff no
    provider-specific prefix detector matches it. Used by /v1/messages to
    decide between the structured OAI tool path, the native Claude passthrough,
    and the legacy text-only soft-fallback.
    """
    model = normalize_model_name(model)
    if is_ollama_model(model):
        return False
    if is_openrouter_model(model):
        return False
    if is_kilo_model(model):
        return False
    if is_codex_model(model):
        return False
    if is_antigravity_model(model):
        return False
    if is_zai_model(model):
        return False
    if is_xiaomi_model(model):
        return False
    if is_opencode_model(model):
        return False
    if is_qwen_model(model):
        return False
    if is_fireworks_model(model):
        return False
    if is_nvidia_model(model):
        return False
    if is_bytedance_model(model):
        return False
    if is_vertex_model(model):
        return False
    if is_lmstudio_model(model):
        return False
    if is_command_code_model(model):
        return False
    if is_command_code_api_model(model):
        return False
    return True


def _make_streaming_gen(
    system_prompt,
    messages,
    model,
    max_tokens,
    oai_messages=None,
    extra_params=None,
    request_id=None,
):
    """Route a model name to its streaming generator.

    ``extra_params`` are threaded through timeout-routed fallbacks so a request
    that disabled reasoning/thinking doesn't silently lose that constraint when
    the proxy switches models.
    """
    model = normalize_model_name(model)
    routed_system_prompt = system_prompt
    routed_messages = messages
    routed_extra_params = extra_params or {}
    if _model_uses_oai_messages(model):
        routed_system_prompt = None
        routed_messages = oai_messages if oai_messages is not None else messages

    # Codex/Antigravity streaming entry points do not accept
    # **extra_params - their reasoning/thinking controls live inside the
    # subprocess config (Codex) or provider-specific request shape. If the
    # caller supplied a non-trivial reasoning/thinking payload and the
    # timeout-routing cascade lands on one of these providers, warn so the
    # operator can diagnose ("why did my enabled reasoning vanish?").
    def _warn_if_dropping_reasoning(provider: str) -> None:
        if not routed_extra_params:
            return
        thinking = routed_extra_params.get("thinking")
        effort = routed_extra_params.get("reasoning_effort")
        reasoning = routed_extra_params.get("reasoning")
        interesting = False
        if isinstance(thinking, dict) and thinking.get("type") not in (
            None,
            "disabled",
        ):
            interesting = True
        if effort is not None:
            interesting = True
        if reasoning is not None:
            interesting = True
        if not interesting:
            return
        rid = f"[{request_id}] " if request_id else ""
        logger.warning(
            f"{rid}fallback {provider} dropping reasoning params "
            f"(thinking={thinking!r}, reasoning={reasoning!r}, reasoning_effort={effort!r}) - "
            f"provider does not plumb extra_params; reasoning will NOT be forwarded"
        )

    if is_ollama_model(model):
        return call_ollama_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_openrouter_model(model):
        return call_openrouter_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_codex_model(model):
        _warn_if_dropping_reasoning("Codex")
        return call_codex_streaming(
            routed_system_prompt, routed_messages, model, max_tokens
        )
    if is_antigravity_model(model):
        _warn_if_dropping_reasoning("Antigravity")
        return call_antigravity_streaming(
            routed_system_prompt, routed_messages, model, max_tokens
        )
    if is_zai_model(model):
        return call_zai_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_xiaomi_model(model):
        return call_xiaomi_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_opencode_model(model):
        return call_opencode_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_qwen_model(model):
        return call_qwen_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_fireworks_model(model):
        return call_fireworks_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_nvidia_model(model):
        return call_nvidia_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_bytedance_model(model):
        return call_bytedance_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_vertex_model(model):
        return call_vertex_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    if is_command_code_api_model(model):
        return call_command_code_api_streaming(
            routed_system_prompt,
            routed_messages,
            model,
            max_tokens,
            **routed_extra_params,
        )
    return call_api_streaming(routed_system_prompt, routed_messages, model, max_tokens)


async def _stream_under_deadline(
    aiter, model, total_deadline, start_time, total_timeout, closer=None
):
    """Iterate an async iterator under a wall-clock deadline, yielding chunks.

    On timeout: calls `closer` (if provided) to release upstream resources, records
    a failure in model_stats so `get_reliable_model()` penalizes this model in future
    fallback selection, and emits a synthetic finish_reason="stop" chunk + [DONE] so
    clients (SkyrimNet) treat this as a natural completion and don't retry. Without
    the synthetic [DONE], clients see a truncated stream and fire a second request -
    causing the "two generations per call" behavior we want to avoid.
    """
    while True:
        try:
            remaining = total_deadline - time.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    pass
            elapsed = time.time() - start_time
            model_stats.record(model, elapsed, success=False)
            logger.warning(
                f"[timeout-routing] {model} stream exceeded total cap {total_timeout}s "
                f"({elapsed:.1f}s) - cutting mid-generation, recorded as failure for smart routing"
            )
            stop_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield chunk


async def _with_timeout_routing_impl(
    gen,
    system_prompt,
    messages,
    oai_messages,
    model,
    max_tokens,
    ttft_timeout=None,
    total_timeout=None,
    extra_params=None,
    _cleanup_streams: Optional[list] = None,
    _closed: Optional[set] = None,
):
    """
    Wrap a streaming generator with two deadlines:

    Phase 1 (before first content token): enforce `ttft_timeout`. If no content arrives
    in time, cancel the stream and re-route to the most reliable fallback model. The total
    deadline also applies here so a hung stream gets the tighter of the two cutoffs.

    Phase 2 (after first content token): enforce `total_timeout` as a hard wall-clock ceiling.
    This cap applies to both the original stream AND any fallback stream kicked off from
    Phase 1, so a slow fallback can't blow past the cap either. If the upstream hasn't
    finished by then, close the stream mid-generation with [DONE]. We can't re-route
    mid-stream because partial content has already been sent to the client.

    Records TTFT to ModelStatsTracker on both success and timeout.
    """
    if ttft_timeout is None:
        ttft_timeout = timeout_cutoff_seconds
    if total_timeout is None:
        total_timeout = max_total_seconds

    aiter = gen.__aiter__()
    start = time.time()
    pre_content_chunks = []

    ttft_deadline = start + ttft_timeout
    total_deadline = start + total_timeout

    # --- Phase 1: wait for first real content chunk (TTFT enforcement + fallback) ---
    while True:
        try:
            phase1_deadline = min(ttft_deadline, total_deadline)
            remaining = phase1_deadline - time.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            for c in pre_content_chunks:
                yield c
            return
        except asyncio.TimeoutError:
            try:
                await gen.aclose()
            except Exception:
                pass
            if _closed is not None:
                _closed.add(id(gen))
            elapsed = time.time() - start
            model_stats.record(model, elapsed, success=False)
            reasoning_only = any(_chunk_has_reasoning(c) for c in pre_content_chunks)
            logger.warning(
                f"[timeout-routing] {model} exceeded pre-content deadline "
                f"({elapsed:.1f}s, TTFT cutoff {ttft_timeout}s / total cap {total_timeout}s)"
                f"{' - reasoning-only, no visible content' if reasoning_only else ''} - cascading to fallbacks"
            )
            # Cascade through reliable fallbacks until one produces content,
            # time runs out, or the candidate pool is exhausted. Before this
            # loop existed we only tried a single fallback and handed it to
            # _stream_under_deadline - if that fallback also failed before
            # producing content, the client got a synthetic empty [DONE] and
            # the game fired a retry (see 2026-04-12 logs: ollama:gemma4 →
            # fireworks:kimi-k2p5-turbo both timed out, game retried with 2 msgs).
            tried: set = {model}
            MAX_CASCADE = 5  # hard cap so a buggy _pick_fallback_model can't loop
            fb_model = None  # last fallback name, for the exhaustion log below
            for _ in range(MAX_CASCADE):
                if time.time() >= total_deadline:
                    break
                fb_model = _pick_fallback_model(exclude=tried)
                if not fb_model or fb_model in tried:
                    break
                tried.add(fb_model)
                logger.warning(f"[timeout-routing] {model} falling back to {fb_model}")

                fb_gen = _make_streaming_gen(
                    system_prompt,
                    messages,
                    fb_model,
                    max_tokens,
                    oai_messages,
                    extra_params=extra_params,
                )
                if _cleanup_streams is not None:
                    _cleanup_streams.append(fb_gen)
                fb_iter = fb_gen.__aiter__()
                fb_start = time.time()
                fb_pre_chunks = []
                fb_got_content = False
                fb_ended_cleanly = False

                # Per-fallback Phase 1: bound the wait for first content by
                # min(ttft_timeout, remaining_total). This is the key change vs.
                # the old path, which gave each fallback the full remaining
                # total budget - one slow fallback could eat all of it.
                fb_ttft_deadline = min(fb_start + ttft_timeout, total_deadline)
                try:
                    while True:
                        rem = fb_ttft_deadline - time.time()
                        if rem <= 0:
                            raise asyncio.TimeoutError()
                        fc = await asyncio.wait_for(fb_iter.__anext__(), timeout=rem)
                        fb_pre_chunks.append(fc)
                        if _chunk_has_content(fc):
                            fb_got_content = True
                            break
                except StopAsyncIteration:
                    # Fallback completed before producing content (empty stream).
                    # Forward what it emitted (role chunks, terminators) so the
                    # client sees something coherent, then stop cascading.
                    fb_ended_cleanly = True
                    try:
                        await fb_gen.aclose()
                    except Exception:
                        pass
                    if _closed is not None:
                        _closed.add(id(fb_gen))
                    model_stats.record(fb_model, time.time() - fb_start, success=False)
                except asyncio.TimeoutError:
                    try:
                        await fb_gen.aclose()
                    except Exception:
                        pass
                    if _closed is not None:
                        _closed.add(id(fb_gen))
                    model_stats.record(fb_model, time.time() - fb_start, success=False)
                    logger.warning(
                        f"[timeout-routing] fallback {fb_model} exceeded TTFT "
                        f"({time.time() - fb_start:.1f}s) - trying next candidate"
                    )
                    continue
                except Exception as e:
                    try:
                        await fb_gen.aclose()
                    except Exception:
                        pass
                    if _closed is not None:
                        _closed.add(id(fb_gen))
                    model_stats.record(fb_model, time.time() - fb_start, success=False)
                    err_msg = str(e) or type(e).__name__
                    logger.error(
                        f"[timeout-routing] fallback {fb_model} errored before first token: {err_msg} - trying next candidate"
                    )
                    continue

                if fb_ended_cleanly:
                    for c in fb_pre_chunks:
                        yield c
                    if not any(
                        isinstance(c, str) and "[DONE]" in c for c in fb_pre_chunks
                    ):
                        for c in _synthetic_stop_done(fb_model, needs_role=not _has_role_chunk(fb_pre_chunks)):
                            yield c
                    return

                # Success: fallback produced real content. Flush its pre-content
                # chunks (role + first content), then stream the remainder under
                # the shared total_deadline so a slow mid-stream fallback still
                # gets hard-cut at max_total_seconds from the ORIGINAL request start.
                if fb_got_content:
                    model_stats.record(fb_model, time.time() - fb_start, success=True)
                    logger.info(
                        f"[timeout-routing] fallback {fb_model} started streaming "
                        f"after {time.time() - fb_start:.1f}s - continuing under total cap"
                    )
                    for c in fb_pre_chunks:
                        yield c
                    try:
                        async for c in _stream_under_deadline(
                            fb_iter,
                            fb_model,
                            total_deadline,
                            fb_start,
                            total_timeout,
                            closer=fb_gen.aclose,
                        ):
                            yield c
                    except Exception as e:
                        err_msg = str(e) or type(e).__name__
                        logger.error(
                            f"[timeout-routing] fallback {fb_model} errored mid-stream: {err_msg}"
                        )
                        # Content already flowing to the client - can't re-route
                        # mid-stream, but we must close cleanly or the game retries.
                        # Use raw stop+[DONE] here (not _synthetic_stop_done_chunks)
                        # because fb_pre_chunks already included the role chunk.
                        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                        _created = int(time.time())
                        stop_only = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": _created,
                            "model": fb_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(stop_only)}\n\n"
                        yield "data: [DONE]\n\n"
                    return

            # Cascade exhausted - no fallback produced content. Emit a synthetic
            # stop+[DONE] so the client treats the call as a (silent) completion
            # instead of retrying. This path is why we track fb_model above:
            # the log should identify which pool we drained.
            logger.error(
                f"[timeout-routing] cascade exhausted (tried {len(tried)} models: "
                f"{sorted(tried)}) - emitting synthetic response"
            )
            # Yield any pre-content chunks (role chunk etc.) collected from
            # the original stream before it timed out - the game expects a
            # role-assistant chunk before any finish_reason chunk.
            for c in pre_content_chunks:
                yield c
            for c in _synthetic_stop_done(model, needs_role=not _has_role_chunk(pre_content_chunks)):
                yield c
            return

        pre_content_chunks.append(chunk)

        # Detect first non-empty content chunk to mark TTFT
        if _chunk_has_content(chunk):
            model_stats.record(model, time.time() - start, success=True)
            break

    for c in pre_content_chunks:
        yield c

    # --- Phase 2: stream remainder under total wall-clock ceiling (hard cut) ---
    # If we blow past the ceiling, the call is marked as a failure in model_stats
    # so get_reliable_model() penalizes this model in future fallback selection -
    # the hard cap becomes a signal for the smart router, not just a hang-guard.
    async for c in _stream_under_deadline(
        aiter, model, total_deadline, start, total_timeout, closer=gen.aclose
    ):
        yield c


async def _with_timeout_routing(
    gen,
    system_prompt,
    messages,
    oai_messages,
    model,
    max_tokens,
    ttft_timeout=None,
    total_timeout=None,
    extra_params=None,
):
    """Own and close every stream created by timeout routing."""
    cleanup_streams = [gen]
    closed = set()
    routed = _with_timeout_routing_impl(
        gen,
        system_prompt,
        messages,
        oai_messages,
        model,
        max_tokens,
        ttft_timeout=ttft_timeout,
        total_timeout=total_timeout,
        extra_params=extra_params,
        _cleanup_streams=cleanup_streams,
        _closed=closed,
    )
    try:
        async for chunk in routed:
            yield chunk
    finally:
        with suppress(Exception):
            await routed.aclose()
        for stream in reversed(cleanup_streams):
            if id(stream) in closed:
                continue
            close = getattr(stream, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()


# --- MITM Interceptor (startup only) ---
# interceptor_handler, _kill_stale_port, start_interceptor,
# recapture_claude_auth, _claude_auth_refresh_loop, codex_interceptor_handler,
# and start_codex_interceptor have been extracted to
# proxy_internal/interceptors.py (Phase 6). They are re-exported via the
# facade imports at the top of this file.


# --- Claude Auth Auto-Refresh ---

_claude_refresh_lock = asyncio.Lock()
_claude_refresh_task: Optional[asyncio.Task] = None


# --- Codex MITM Interceptor (startup only) ---

CODEX_INTERCEPTOR_PORT = 9998


# --- Direct API call ---


# --- Z.AI ---

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


# --- Xiaomi ---

XIAOMI_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
XIAOMI_PLATFORM_URL = "https://api.xiaomimimo.com/v1"
# Models only available on the platform endpoint, not the SGP token-plan
_XIAOMI_PLATFORM_MODELS = {"mimo-v2-flash", "mimo-v2.5-asr", "mimo-v2.5-tts", "mimo-v2.5-tts-voiceclone", "mimo-v2.5-tts-voicedesign", "mimo-v2-tts"}


# --- OpenCode (Zen + Go APIs - OpenAI-compatible) ---


# --- Qwen Code (OpenAI-compatible via portal.qwen.ai) ---


# --- Fireworks (OpenAI-compatible via api.fireworks.ai) ---

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


# --- Antigravity Auth Loading ---

ANTIGRAVITY_AUTH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "antigravity-auth.json"
)


# --- OpenCode Auth Loading ---
# load_opencode_key has been extracted to proxy_internal/interceptors.py
# (Phase 6) and re-exported via the facade imports at the top of this file.


# --- OpenAI-compatible API ---


@asynccontextmanager
async def lifespan(app):
    global ollama_session, lmstudio_session, third_party_session

    # Load per-provider settings (enabled/timeout_ms/priority) into the in-memory
    # cache now that the registry module is safe to import. Must happen before
    # anything reads provider_config._cache (health_fn wrappers, listing tools).
    from proxy_internal import provider_config

    provider_config.reload_from_config()

    # Start interceptors - failures should not block other providers
    claude_runner = None
    codex_runner = None

    try:
        claude_runner = await start_interceptor()
    except Exception as e:
        logger.warning(f"Claude auth capture failed: {e}. Claude provider unavailable.")

    try:
        codex_runner = await start_codex_interceptor()
    except Exception as e:
        logger.warning(f"Codex auth capture failed: {e}. Codex provider unavailable.")

    # Persistent Ollama session - shared across all Ollama requests
    ollama_session = create_session()
    lmstudio_session = create_session()
    third_party_session = create_session()

    # macOS uses agy's native session; other platforms retain legacy OAuth.
    await load_antigravity_auth()
    _log_auth_source("Antigravity", antigravity_auth_source(),
                     "sign in with agy" if antigravity_auth.native_keychain else "visit /config/antigravity-login")
    # Load OpenCode API key from auth file (if not in config.json)
    load_opencode_key()
    # Load Qwen Code auth from credentials file
    await load_qwen_auth()
    # Start background auth refresh loop for Claude
    global _claude_refresh_task
    if auth.is_ready and CLAUDE_PATH:
        _claude_refresh_task = asyncio.create_task(_claude_auth_refresh_loop())
        logger.info("Claude auth auto-refresh enabled (every 45 min)")

    # Start background stats persistence loop (flushes to stats.json every 60s)
    global _stats_persist_task
    _stats_persist_task = asyncio.create_task(_stats_persist_loop())
    logger.info(
        f"Stats persistence enabled (flushing to {STATS_FILE} every {int(_stats_persist_interval_s)}s)"
    )

    # Freeze all module-level objects so the GC skips scanning them on every cycle.
    gc.freeze()
    yield
    # Cancel background refresh task
    if _claude_refresh_task and not _claude_refresh_task.done():
        _claude_refresh_task.cancel()
    # Cancel stats persist loop and do a final flush before shutdown
    if _stats_persist_task and not _stats_persist_task.done():
        _stats_persist_task.cancel()
    try:
        payload = _snapshot_stats()
        await asyncio.to_thread(_write_stats_to_disk, payload)
        logger.info("Final stats flush complete")
    except Exception as e:
        logger.warning(f"Final stats flush failed: {e}")
    if auth.session:
        await auth.session.close()
    if ollama_session:
        await ollama_session.close()
    if lmstudio_session:
        await lmstudio_session.close()
    if third_party_session:
        await third_party_session.close()
    if codex_auth.session:
        await codex_auth.session.close()
    if qwen_auth.session:
        await qwen_auth.session.close()
    # Close all Antigravity account sessions
    for account in antigravity_auth.accounts:
        if account.session:
            await account.session.close()
    if antigravity_auth._legacy_account.session:
        await antigravity_auth._legacy_account.session.close()
    await close_vertex_clients()
    if claude_runner:
        await claude_runner.cleanup()
    if codex_runner:
        await codex_runner.cleanup()


# Build CORS origins - localhost always, plus Tailscale FQDN if available
_cors_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8539",
    "http://127.0.0.1:8539",
    "http://localhost:8432",
    "http://127.0.0.1:8432",
]
_ts_fqdn = get_tailscale_fqdn()
if _ts_fqdn:
    for _port in (8000, 8539, 8432):
        _cors_origins.append(f"https://{_ts_fqdn}:{_port}")
    logger.info(f"Tailscale CORS origins added for {_ts_fqdn}")

app = FastAPI(title="Claude SkyrimNet Proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _config_csrf_guard(request: Request, call_next):
    if request.method in {
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    } and request.url.path.startswith("/config/"):
        if request.cookies.get(ADMIN_CSRF_COOKIE) != ADMIN_CSRF_TOKEN:
            return JSONResponse(
                status_code=403, content={"detail": "Dashboard session required"}
            )
    return await call_next(request)


# --- Register route modules (Phase 5) ---
from proxy_internal.endpoints.health import router as _health_router
from proxy_internal.endpoints.stats_endpoint import router as _stats_router
from proxy_internal.endpoints.config_keys import router as _config_keys_router
from proxy_internal.endpoints.config_runtime import router as _config_runtime_router
from proxy_internal.endpoints.models import router as _models_router
from proxy_internal.endpoints.antigravity_oauth import (
    router as _antigravity_oauth_router,
)
from proxy_internal.endpoints.anthropic_messages import router as _anthropic_router
from proxy_internal.endpoints.chat_completions import router as _chat_router
from proxy_internal.endpoints.dashboard import router as _dashboard_router
from proxy_internal.endpoints.config_providers import router as _config_providers_router

app.include_router(_health_router)
app.include_router(_stats_router)
app.include_router(_config_keys_router)
app.include_router(_config_runtime_router)
app.include_router(_models_router)
app.include_router(_antigravity_oauth_router)
app.include_router(_anthropic_router)
app.include_router(_chat_router)
app.include_router(_dashboard_router)
app.include_router(_config_providers_router)

_STREAMING_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _valid_opencode_session_id(value) -> bool:
    """Conservative printable token 1..128 chars for inbound x-opencode-session."""
    import re
    return isinstance(value, str) and re.fullmatch(r"[\x20-\x7E]{1,128}", value) is not None


async def _chat_completions_inner(req: ChatRequest, opencode_session_id=None):
    # Parse model list and pick via round-robin
    model_field = req.model or DEFAULT_MODEL
    models = parse_model_list(model_field)
    model = normalize_model_name(
        pick_model_round_robin(models) if len(models) > 1 else models[0]
    )
    use_ollama = is_ollama_model(model)
    use_lmstudio = is_lmstudio_model(model)
    use_openrouter = not use_ollama and not use_lmstudio and is_openrouter_model(model)
    use_kilo = not use_ollama and not use_lmstudio and is_kilo_model(model)
    use_codex = not use_ollama and not use_lmstudio and is_codex_model(model)
    use_antigravity = (
        not use_ollama and not use_lmstudio and is_antigravity_model(model)
    )
    use_zai = not use_ollama and not use_lmstudio and is_zai_model(model)
    use_xiaomi = not use_ollama and not use_lmstudio and is_xiaomi_model(model)
    use_opencode = not use_ollama and not use_lmstudio and is_opencode_model(model)
    use_qwen = not use_ollama and not use_lmstudio and is_qwen_model(model)
    use_fireworks = not use_ollama and not use_lmstudio and is_fireworks_model(model)
    use_nvidia = not use_ollama and not use_lmstudio and is_nvidia_model(model)
    use_bytedance = not use_ollama and not use_lmstudio and is_bytedance_model(model)
    use_vertex = not use_ollama and not use_lmstudio and is_vertex_model(model)
    use_command_code = not use_ollama and not use_lmstudio and is_command_code_model(model)
    use_command_api = not use_ollama and not use_lmstudio and is_command_code_api_model(model)
    command_api_key = None

    if use_ollama:
        pass  # No auth check - local needs none, cloud key checked inside call function
    elif use_lmstudio:
        pass  # No auth check - local only
    elif use_openrouter:
        pass  # No auth check needed for OpenRouter
    elif use_kilo:
        if not kilo_api_key:
            raise HTTPException(
                status_code=503,
                detail="Kilo Gateway API key not configured - add it via /config/kilo-key",
            )
    elif use_zai:
        if not zai_api_key:
            raise HTTPException(
                status_code=503,
                detail="Z.AI API key not configured - add it via /config/zai-key",
            )
    elif use_xiaomi:
        if not xiaomi_api_key:
            raise HTTPException(
                status_code=503,
                detail="Xiaomi API key not configured - add it via /config/xiaomi-key",
            )
    elif use_opencode:
        _, _, resolved_key, plan = _resolve_opencode(model)
        if not resolved_key:
            raise HTTPException(
                status_code=503,
                detail=f"OpenCode {plan} API key not configured - add it via /config/opencode-key",
            )
    elif use_fireworks:
        if not fireworks_api_key:
            raise HTTPException(
                status_code=503,
                detail="Fireworks API key not configured - add it via /config/fireworks-key",
            )
    elif use_nvidia:
        if not nvidia_api_key:
            raise HTTPException(
                status_code=503,
                detail="NVIDIA NIM API key not configured - add it via /config/nvidia-key",
            )
    elif use_bytedance:
        if not bytedance_api_key:
            raise HTTPException(
                status_code=503,
                detail="ByteDance API key not configured - add it via /config/bytedance-key",
            )
    elif use_vertex:
        if not _resolve_vertex_api_key():
            raise HTTPException(
                status_code=503,
                detail="Vertex AI API key not configured - add it via /config/vertex-key",
            )
    elif use_command_api:
        command_api_key = _resolve_command_code_api_key()
        if not command_api_key:
            raise HTTPException(
                status_code=503,
                detail="Command Code API key not configured - set COMMAND_CODE_API_KEY or add command_code_api_key via /config",
            )
    elif use_command_code:
        if not COMMAND_CODE_PATH:
            raise HTTPException(
                status_code=503,
                detail="Command Code CLI not installed — install from commandcode.ai and run 'cmd login'",
            )
    elif use_qwen:
        if not qwen_auth.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Qwen auth not ready -- run Qwen Code CLI login first",
            )
    elif use_codex:
        if not codex_auth.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Codex auth not ready -- ensure you have run 'codex login'",
            )
    elif use_antigravity:
        if antigravity_auth.native_keychain and not antigravity_auth.is_ready:
            await antigravity_auth.refresh_if_needed()
        if not antigravity_auth.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Antigravity auth not ready -- sign in with agy"
                if antigravity_auth.native_keychain else "Antigravity auth not ready -- visit /config/antigravity-login",
            )
    else:
        if not auth.is_ready:
            raise HTTPException(
                status_code=503, detail="Claude auth not ready -- warming up"
            )

    system_prompt, merged, oai_messages = _normalize_chat_messages(req.messages)

    if not merged:
        raise HTTPException(status_code=400, detail="No user message provided")

    max_tokens = req.max_tokens or 100000

    # Whitelist extra params - only forward standard OpenAI sampling parameters.
    # SkyrimNet sends provider-specific fields (provider, reasoning, etc.) that
    # cause 401/400 errors on providers that don't understand them.
    extra_params = {
        k: v
        for k, v in (req.model_extra or {}).items()
        if v is not None and k in _CHAT_ALLOWED_EXTRA
    }
    # Declared-on-ChatRequest sampling fields live outside model_extra, so
    # merge them in explicitly - otherwise top-level `temperature` from the
    # request body would be silently dropped.
    if req.temperature is not None:
        extra_params.setdefault("temperature", req.temperature)

    # Thinking/reasoning precedence (highest first):
    #   1. Dashboard reasoning override ON → force-enable at configured level,
    #      overriding whatever the caller sent. This is how CCS-routed Claude
    #      Code gets reasoning back on: CCS's per-tier thinking defaults only
    #      apply to its own cliproxy providers, so every CCS request hits us
    #      with ``thinking`` absent and would otherwise be silenced by (3).
    #   2. Caller supplied ``thinking`` → preserve verbatim. Legacy behavior
    #      for explicit SkyrimNet / game callers that know what they want.
    #   3. Otherwise → default to disabled. Without this, reasoning models
    #      (kimi-k2, mimo-v2, minimax, etc.) produce chain-of-thought tokens
    #      that get scrubbed server-side, leaving 0-char responses.
    if reasoning_override_enabled:
        extra_params["thinking"] = _override_thinking_payload(max_tokens)
        # OAI providers (OpenRouter, Ollama, Fireworks, etc.) look for
        # ``reasoning_effort``; Anthropic providers read ``thinking``. Inject
        # both so whichever field the downstream provider honours is set.
        extra_params["reasoning_effort"] = reasoning_override_level
    else:
        _caller_thinking = extra_params.get("thinking")
        if _caller_thinking is None:
            # In MCP mode, leave thinking unset so providers use their defaults
            # (reasoning enabled). Only force-disabled in proxy mode where
            # reasoning tokens get scrubbed and produce empty NPC responses.
            if not MCP_MODE:
                extra_params.setdefault("thinking", {"type": "disabled"})

    def _stream(gen):
        """Wrap a streaming generator with timeout routing if enabled."""
        if timeout_routing_enabled:
            return _with_timeout_routing(
                gen,
                system_prompt,
                merged,
                oai_messages,
                model,
                max_tokens,
                extra_params=extra_params,
            )
        return gen

    async def _tracked_stream(gen):
        """Wrap a streaming generator to record stats."""
        start = time.time()
        had_error = True  # assume error until we see a clean finish
        try:
            async for chunk in gen:
                yield chunk
                if isinstance(chunk, str) and "[DONE]" in chunk:
                    had_error = False
        except Exception:
            raise
        finally:
            close = getattr(gen, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()
            request_stats.record(model, "streaming", time.time() - start, not had_error)

    async def _call_direct(coro):
        """Await a direct call and record stats. Direct (non-streaming) calls are
        NOT bounded by timeout routing - summary/memory workflows legitimately take
        longer than the streaming TTS budget."""
        start = time.time()
        try:
            result = await coro
            request_stats.record(model, "direct", time.time() - start, True)
            return result
        except Exception:
            request_stats.record(model, "direct", time.time() - start, False)
            raise

    # Route to correct provider using the booleans already computed above.
    if use_ollama:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_ollama_direct,
            call_ollama_streaming,
            True,
            "ollama",
        )
    elif use_lmstudio:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_lmstudio_direct,
            call_lmstudio_streaming,
            True,
            "lmstudio",
        )
    elif use_openrouter:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_openrouter_direct,
            call_openrouter_streaming,
            True,
            "openrouter",
        )
    elif use_kilo:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_kilo_direct,
            call_kilo_streaming,
            True,
            "kilo",
        )
    elif use_codex:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_codex_direct,
            call_codex_streaming,
            False,
            "codex_cli",
        )
    elif use_antigravity:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_antigravity_direct,
            call_antigravity_streaming,
            False,
            "antigravity",
        )
    elif use_zai:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_zai_direct,
            call_zai_streaming,
            True,
            "zai",
        )
    elif use_xiaomi:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_xiaomi_direct,
            call_xiaomi_streaming,
            True,
            "xiaomi",
        )
    elif use_opencode:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_opencode_direct,
            call_opencode_streaming,
            True,
            # Zen and Go share these call functions; resolved per model below.
            "opencode_zen",
        )
    elif use_qwen:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_qwen_direct,
            call_qwen_streaming,
            True,
            "qwen_cli",
        )
    elif use_fireworks:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_fireworks_direct,
            call_fireworks_streaming,
            True,
            "fireworks",
        )
    elif use_nvidia:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_nvidia_direct,
            call_nvidia_streaming,
            True,
            "nvidia",
        )
    elif use_bytedance:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_bytedance_direct,
            call_bytedance_streaming,
            True,
            "bytedance",
        )
    elif use_vertex:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_vertex_direct,
            call_vertex_streaming,
            True,
            "vertex",
        )
    elif use_command_api:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_command_code_api_direct,
            call_command_code_api_streaming,
            True,
            "command_api",
        )
    elif use_command_code:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_command_code_direct,
            call_command_code_streaming,
            True,
            "command_code",
        )
    else:
        direct_fn, stream_fn, use_extra, provider_name = (
            call_api_direct,
            call_api_streaming,
            False,
            "claude",
        )

    call_system_prompt = None if _model_uses_oai_messages(model) else system_prompt
    call_messages = oai_messages if _model_uses_oai_messages(model) else merged
    if is_opencode_go_model(model) and _valid_opencode_session_id(opencode_session_id):
        extra_params["opencode_session_id"] = opencode_session_id
    kwargs = extra_params if use_extra else {}
    if use_command_api:
        kwargs["api_key"] = command_api_key

    # Warn when reasoning params will be dropped by a provider that does not
    # accept extra_params at all. This covers both the dashboard override and
    # caller-supplied reasoning_effort/thinking requests.
    if not use_extra and (use_codex or use_antigravity):
        thinking = extra_params.get("thinking")
        effort = extra_params.get("reasoning_effort")
        reasoning = extra_params.get("reasoning")
        interesting = (
            (
                isinstance(thinking, dict)
                and thinking.get("type") not in (None, "disabled")
            )
            or effort is not None
            or reasoning is not None
        )
        if interesting:
            logger.warning(
                f"provider {'Codex' if use_codex else 'Antigravity'} "
                f"dropping reasoning params (thinking={thinking!r}, reasoning={reasoning!r}, reasoning_effort={effort!r}) - "
                f"provider does not plumb extra_params; reasoning will NOT be forwarded"
            )

    # TODO: Add off/tags/voice/all sanitizer modes and bypass structured/tool output.
    output_request_id = uuid.uuid4().hex[:8]

    # Hold one slot for this provider for the whole call — that pool is what keeps
    # plan-wide concurrency caps (z.ai's 6 simultaneous calls, for one) from
    # turning into 429s. Providers with no configured limit get an inert slot.
    if use_opencode and is_opencode_go_model(model):
        provider_name = "opencode_go"
    slot = await concurrency.acquire(provider_name, request_id=output_request_id)

    if req.stream:
        routed_stream = _stream(
            concurrency.hold_slot(
                stream_fn(
                    call_system_prompt,
                    call_messages,
                    model,
                    max_tokens,
                    **kwargs,
                ),
                slot,
            )
        )

        async def _stream_with_rewrite_context(events):
            iterator = events.__aiter__()
            try:
                while True:
                    token = (
                        _rewrite_ctx.set({"system_prompt": system_prompt, "model": model})
                        if not MCP_MODE
                        else None
                    )
                    try:
                        try:
                            chunk = await iterator.__anext__()
                        except StopAsyncIteration:
                            return
                    finally:
                        if token is not None:
                            _rewrite_ctx.reset(token)
                    yield chunk
            finally:
                await _close_async_iterator(iterator)

        contextual_stream = _stream_with_rewrite_context(routed_stream)
        sanitized_stream = sanitize_sse_stream(
            contextual_stream, output_request_id, model
        )
        prepared_stream = await preflight_sse_stream(sanitized_stream)
        return StreamingResponse(
            _tracked_stream(prepared_stream),
            media_type="text/event-stream",
            headers=_STREAMING_HEADERS,
        )
    try:
        response = await _call_direct(
            direct_fn(call_system_prompt, call_messages, model, max_tokens, **kwargs)
        )
    finally:
        slot.release()

    if response is None:
        raise HTTPException(status_code=500, detail="Missing response")
    response = sanitize_response(response, output_request_id, model)

    prompt_chars = sum(len(msg.content) for msg in req.messages)
    prompt_chars += max(0, len(req.messages) - 1)
    prompt_tokens = prompt_chars // 4
    completion_tokens = len(response) // 4

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response, "name": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    }


# ---------------------------------------------------------------------------
# Anthropic Messages API compatibility layer (/v1/messages)
#
# Translates the Anthropic Messages API shape into the existing OpenAI-format
# pipeline and back.  Scope: text in / text out.  Tool definitions in the
# request are flattened into the system prompt so the backend model is at
# least aware they exist; tool_use / tool_result blocks in message history are
# rendered into plain text so conversational context survives.  The backend
# providers in this proxy do not emit structured tool_calls, so the assistant
# cannot fire tools end-to-end - clients like Claude Code that require tool
# execution should use this endpoint only for chat-style interactions.
# ---------------------------------------------------------------------------

# Anthropic stop_reason <- OpenAI finish_reason
_ANTHROPIC_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "stop_sequence",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def _anthropic_flatten_system(system_field) -> Optional[str]:
    """Anthropic 'system' may be a string or a list of content blocks."""
    if system_field is None:
        return None
    if isinstance(system_field, str):
        return system_field or None
    if isinstance(system_field, list):
        parts = []
        for block in system_field:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(parts) or None
    return None


def _anthropic_flatten_content(content) -> str:
    """Render an Anthropic message content field as plain text.

    Content can be a string or a list of content blocks.  Tool_use / tool_result
    blocks are rendered in a readable form so conversation context survives the
    trip through a text-only pipeline.  Image blocks collapse to a placeholder.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            if block:
                parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "")
            tool_input = block.get("input", {})
            try:
                rendered_input = json.dumps(tool_input, ensure_ascii=False)
            except (TypeError, ValueError):
                rendered_input = str(tool_input)
            parts.append(f"[tool_use name={name} input={rendered_input}]")
        elif btype == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner_text = _anthropic_flatten_content(inner)
            else:
                inner_text = str(inner) if inner is not None else ""
            is_error = " error" if block.get("is_error") else ""
            parts.append(f"[tool_result id={tool_use_id}{is_error}]\n{inner_text}")
        elif btype == "image":
            parts.append("[image]")
        elif btype == "thinking":
            # Extended thinking blocks are internal - drop silently
            continue
        elif btype == "document":
            parts.append("[document]")
    return "\n\n".join(p for p in parts if p)


def _anthropic_tools_to_system_hint(tools) -> Optional[str]:
    """Render Anthropic tool definitions as a system-prompt hint.

    The backend text-only pipeline cannot natively dispatch tools, but including
    the tool catalogue in the system prompt means the model at least knows what
    capabilities exist in the conversation.
    """
    if not tools or not isinstance(tools, list):
        return None
    summaries = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        if not name:
            continue
        desc = (tool.get("description") or "").strip()
        if desc:
            summaries.append(f"- {name}: {desc}")
        else:
            summaries.append(f"- {name}")
    if not summaries:
        return None
    return "Available tools (text-only description):\n" + "\n".join(summaries)


def _anthropic_request_to_chat_request(body: dict) -> "ChatRequest":
    """Translate an Anthropic /v1/messages payload to this proxy's ChatRequest."""
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )

    model = body.get("model") or DEFAULT_MODEL
    raw_messages = body.get("messages") or []
    if not isinstance(raw_messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be an array")

    system_prompt = _anthropic_flatten_system(body.get("system"))
    tool_hint = _anthropic_tools_to_system_hint(body.get("tools"))
    if tool_hint:
        system_prompt = (
            f"{system_prompt}\n\n{tool_hint}" if system_prompt else tool_hint
        )

    flat_messages: list[dict] = []
    if system_prompt:
        flat_messages.append({"role": "system", "content": system_prompt})

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _anthropic_flatten_content(msg.get("content"))
        if not text:
            continue
        flat_messages.append({"role": role, "content": text})

    if not any(m["role"] == "user" for m in flat_messages):
        raise HTTPException(
            status_code=400, detail="At least one user message is required"
        )

    # Build a ChatRequest.  Only forward sampling params the existing pipeline
    # already whitelists via _CHAT_ALLOWED_EXTRA.
    chat_payload: dict = {
        "model": model,
        "messages": flat_messages,
        "max_tokens": body.get("max_tokens") or 100000,
        "stream": bool(body.get("stream")),
    }
    if "temperature" in body and body["temperature"] is not None:
        chat_payload["temperature"] = body["temperature"]
    if "top_p" in body and body["top_p"] is not None:
        chat_payload["top_p"] = body["top_p"]
    if "top_k" in body and body["top_k"] is not None:
        chat_payload["top_k"] = body["top_k"]
    stop_sequences = body.get("stop_sequences")
    if stop_sequences:
        chat_payload["stop"] = stop_sequences

    return ChatRequest(**chat_payload)


# ---------------------------------------------------------------------------
# Anthropic /v1/messages - structured tool_use passthrough path
#
# The text-only translator above flattens tool_use/tool_result blocks into
# plain text so the backend sees a conversational summary.  For clients that
# require the assistant to actually fire tools (Claude Code, Anthropic SDK
# agents), the helpers below offer a separate pipeline that routes directly
# to OpenAI-compatible providers and preserves structured tool calls on both
# the request and response sides.  Antigravity participates via a Gemini-
# format adapter that translates between OAI and Gemini wire protocols.
# Non-OAI-compatible providers without an adapter (Claude native, Codex CLI)
# cannot participate because their upstream response format does
# not emit OpenAI-style tool_calls.
# ---------------------------------------------------------------------------

_OAI_COMPATIBLE_PROVIDERS = (
    "OpenRouter",
    "Ollama",
    "Z.AI",
    "Xiaomi",
    "OpenCode",
    "Qwen",
    "Fireworks",
    "NVIDIA NIM",
    "Antigravity",
)


async def _resolve_oai_compatible(model: str) -> Optional[dict]:
    """Resolve an OpenAI-compatible provider for the given model name.

    Returns a dict with the endpoint, api_model, headers, aiohttp session, and
    an ``owns_session`` flag indicating whether the caller must close the
    session after use.  Returns ``None`` when the model does not belong to any
    OpenAI-compatible provider (e.g. ``claude-sonnet-4-6``).

    This coroutine is async because the Qwen branch must go through
    ``qwen_auth.refresh_if_needed()`` (which takes an ``asyncio.Lock``) rather
    than calling ``reload_from_file`` directly - otherwise two concurrent
    ``/v1/messages`` tool requests could race on the shared token state.
    """
    if is_ollama_model(model):
        api_model = model[len("ollama:"):]
        _sync_discovered_api_keys()
        if ollama_api_key:
            endpoint = "https://ollama.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {ollama_api_key}",
                "Content-Type": "application/json",
            }
        else:
            endpoint = "http://localhost:11434/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        session = ollama_session or auth.session or create_session()
        owns_session = not ollama_session and not auth.session
        return {
            "endpoint_url": endpoint,
            "api_model": api_model,
            "headers": headers,
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Ollama",
        }

    if is_openrouter_model(model):
        _sync_discovered_api_keys()
        if not openrouter_api_key:
            raise HTTPException(
                status_code=503, detail="OpenRouter API key not configured"
            )
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": "https://openrouter.ai/api/v1/chat/completions",
            "api_model": model,
            "headers": {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "OpenRouter",
        }

    if is_kilo_model(model):
        if not kilo_api_key:
            raise HTTPException(
                status_code=503, detail="Kilo Gateway API key not configured"
            )
        api_model = _resolve_kilo_model(model)
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": "https://api.kilo.ai/api/gateway/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {kilo_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Kilo Gateway",
        }

    if is_zai_model(model):
        if not zai_api_key:
            raise HTTPException(status_code=503, detail="Z.AI API key not configured")
        api_model = model[len("zai:"):]
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": f"{ZAI_BASE_URL}/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {zai_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Z.AI",
        }

    if is_xiaomi_model(model):
        if not xiaomi_api_key:
            raise HTTPException(status_code=503, detail="Xiaomi API key not configured")
        api_model = model[len("xiaomi:"):]
        base = (
            XIAOMI_PLATFORM_URL
            if api_model in _XIAOMI_PLATFORM_MODELS
            else XIAOMI_BASE_URL
        )
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": f"{base}/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {xiaomi_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Xiaomi",
        }

    if is_opencode_model(model):
        api_model, base_url, api_key, plan = _resolve_opencode(model)
        if not api_key:
            raise HTTPException(
                status_code=503, detail=f"OpenCode {plan} API key not configured"
            )
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": f"{base_url}/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "opencode/1.3.10",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": f"OpenCode {plan}",
        }

    if is_qwen_model(model):
        # Go through refresh_if_needed so the per-instance asyncio.Lock
        # serialises concurrent token reloads.  Bypassing it (as the old
        # direct reload_from_file() call did) races two coroutines on
        # qwen_auth.access_token / .expires_at.
        await qwen_auth.refresh_if_needed()
        if not qwen_auth.is_ready:
            raise HTTPException(status_code=503, detail="Qwen auth not ready")
        api_model = model[len("qwen:"):]
        session = qwen_auth.session or create_session()
        owns_session = not qwen_auth.session
        return {
            "endpoint_url": f"{QWEN_BASE_URL}/chat/completions",
            "api_model": api_model,
            "headers": qwen_auth.get_auth_headers(),
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Qwen",
        }

    if is_fireworks_model(model):
        if not fireworks_api_key:
            raise HTTPException(
                status_code=503, detail="Fireworks API key not configured"
            )
        api_model = _resolve_fireworks_model(model)
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": f"{FIREWORKS_BASE_URL}/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {fireworks_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Fireworks",
        }

    if is_nvidia_model(model):
        if not nvidia_api_key:
            raise HTTPException(
                status_code=503, detail="NVIDIA NIM API key not configured"
            )
        api_model = _resolve_nvidia_model(model)
        session = third_party_session or auth.session or create_session()
        owns_session = (
            session is not third_party_session and session is not auth.session
        )
        return {
            "endpoint_url": f"{NVIDIA_BASE_URL}/chat/completions",
            "api_model": api_model,
            "headers": {
                "Authorization": f"Bearer {nvidia_api_key}",
                "Content-Type": "application/json",
            },
            "session": session,
            "owns_session": owns_session,
            "provider_name": "NVIDIA NIM",
        }

    # Antigravity uses the Gemini wire protocol (generateContent / streamGenerateContent).
    # The resolved dict carries format="gemini" so _anthropic_messages_structured
    # can dispatch to the Antigravity-specific adapter instead of the generic OAI path.
    if is_antigravity_model(model):
        if not antigravity_auth.is_ready and not antigravity_auth.native_keychain:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready")
        refresh_result = antigravity_auth.refresh_if_needed()
        if inspect.isawaitable(refresh_result):
            await refresh_result
        if not antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready")
        api_model = _get_antigravity_model_id(model)
        account = antigravity_auth.get_next_account()
        if not account:
            raise HTTPException(
                status_code=503, detail="No Antigravity accounts available"
            )
        session = account.session or antigravity_auth.session or create_session()
        owns_session = not account.session and not antigravity_auth.session
        return {
            "endpoint_url": f"{ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
            "stream_url": f"{ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse",
            "api_model": api_model,
            "headers": account.get_auth_headers(),
            "session": session,
            "owns_session": owns_session,
            "provider_name": "Antigravity",
            "format": "gemini",
            "project_id": account.project_id or ANTIGRAVITY_DEFAULT_PROJECT_ID,
        }

    return None


def _anthropic_content_blocks_to_plain_text(content) -> str:
    """Flatten a tool_result content field (string or block list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
                else:
                    try:
                        parts.append(json.dumps(block))
                    except (TypeError, ValueError):
                        parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _anthropic_tools_to_oai(tools) -> list[dict]:
    """Translate Anthropic tool defs to OpenAI function tool defs."""
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="'tools' must be a list")
    oai_tools: list[dict] = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise HTTPException(status_code=400, detail=f"tools[{i}] must be an object")
        name = t.get("name")
        if not name or not isinstance(name, str):
            raise HTTPException(status_code=400, detail=f"tools[{i}] missing 'name'")
        description = t.get("description") or ""
        schema = (
            t.get("input_schema")
            or t.get("parameters")
            or {"type": "object", "properties": {}}
        )
        if not isinstance(schema, dict):
            raise HTTPException(
                status_code=400, detail=f"tools[{i}] input_schema must be an object"
            )
        oai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return oai_tools


def _anthropic_tool_choice_to_oai(tool_choice):
    """Translate Anthropic tool_choice to OpenAI tool_choice."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str) and tool_choice in ("auto", "none", "required"):
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise HTTPException(
            status_code=400, detail="tool_choice must be an object or string"
        )
    t = tool_choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        name = tool_choice.get("name")
        if not name:
            raise HTTPException(
                status_code=400, detail="tool_choice type=tool requires 'name'"
            )
        return {"type": "function", "function": {"name": name}}
    raise HTTPException(status_code=400, detail=f"Unsupported tool_choice type: {t!r}")


def _anthropic_messages_to_oai_structured(
    messages: list,
) -> tuple[list[dict], list[str]]:
    """Translate Anthropic-format messages into OpenAI chat messages for the
    structured tool_use passthrough path.

    Rules:
      - ``tool_use`` blocks in an assistant message become entries in that
        message's ``tool_calls`` array (with content=None when there was no
        accompanying text).
      - ``tool_result`` blocks in a user message become separate
        ``{role: "tool", tool_call_id, content}`` messages immediately after
        any text from the same user message.
      - Plain ``text`` blocks are concatenated.
      - ``image`` blocks are collapsed to a text placeholder (this path does
        not claim vision parity).
      - Unknown block types raise HTTP 400.

    Returns a tuple ``(oai_messages, in_array_system_texts)``.  In-array
    ``{role: "system"}`` entries are NOT inserted into ``oai_messages`` - they
    are collected into ``in_array_system_texts`` so the caller can merge them
    with any top-level ``system`` field into a **single** system message at
    index 0 (preventing duplicate system messages that several upstreams
    reject).
    """
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")

    oai: list[dict] = []
    in_array_system_texts: list[str] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise HTTPException(
                status_code=400, detail=f"messages[{i}] must be an object"
            )
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant", "system"):
            raise HTTPException(
                status_code=400, detail=f"messages[{i}] has unknown role {role!r}"
            )

        if role == "system":
            text = _anthropic_content_blocks_to_plain_text(content)
            if text:
                in_array_system_texts.append(text)
            continue

        if isinstance(content, str):
            oai.append({"role": role, "content": content})
            continue
        if content is None:
            oai.append({"role": role, "content": ""})
            continue
        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{i}].content must be string or list",
            )

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}].content[{j}] must be an object",
                )
            btype = block.get("type")
            if btype == "text":
                if block.get("text"):
                    text_parts.append(str(block["text"]))
            elif btype == "tool_use":
                tu_id = block.get("id")
                tu_name = block.get("name")
                tu_input = block.get("input", {})
                if not tu_id or not tu_name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"messages[{i}].content[{j}] tool_use missing id/name",
                    )
                try:
                    arg_str = (
                        json.dumps(tu_input)
                        if not isinstance(tu_input, str)
                        else tu_input
                    )
                except (TypeError, ValueError):
                    arg_str = "{}"
                tool_calls.append(
                    {
                        "id": tu_id,
                        "type": "function",
                        "function": {"name": tu_name, "arguments": arg_str},
                    }
                )
            elif btype == "tool_result":
                tr_id = block.get("tool_use_id")
                if not tr_id:
                    raise HTTPException(
                        status_code=400,
                        detail=f"messages[{i}].content[{j}] tool_result missing tool_use_id",
                    )
                tr_text = _anthropic_content_blocks_to_plain_text(block.get("content"))
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr_id,
                        "content": tr_text,
                    }
                )
            elif btype == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    media = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    text_parts.append(f"[image: data:{media};base64,{len(data)}B]")
                elif src.get("type") == "url":
                    text_parts.append(f"[image: {src.get('url', '')}]")
            elif btype == "thinking":
                continue
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}].content[{j}] unknown type {btype!r}",
                )

        if role == "assistant":
            out: dict = {"role": "assistant"}
            if text_parts:
                out["content"] = "\n".join(text_parts)
            else:
                out["content"] = None if tool_calls else ""
            if tool_calls:
                out["tool_calls"] = tool_calls
            oai.append(out)
        else:  # user
            if text_parts:
                oai.append({"role": "user", "content": "\n".join(text_parts)})
            oai.extend(tool_results)
            if not text_parts and not tool_results:
                oai.append({"role": "user", "content": ""})

    return oai, in_array_system_texts


def _anthropic_to_oai_structured(body: dict) -> dict:
    """Translate an Anthropic /v1/messages body into a structured OpenAI payload.

    Preserves tool defs, tool_choice, tool_use blocks, and tool_result blocks
    end-to-end so OpenAI-compatible upstreams can dispatch tools natively.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")

    model = body.get("model")
    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="'model' is required")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            status_code=400, detail="'messages' required and must be non-empty"
        )

    oai_messages, in_array_system_texts = _anthropic_messages_to_oai_structured(
        messages
    )

    # Merge the top-level ``system`` field with any in-array system messages
    # into EXACTLY ONE ``{role: "system"}`` entry at index 0.  Several OpenAI-
    # compatible upstreams reject requests that contain more than one
    # consecutive system message, so we collapse them here regardless of which
    # combination the caller sent (top-level only, in-array only, or both).
    top_level_system = _anthropic_flatten_system(body.get("system"))
    system_parts: list[str] = []
    if top_level_system:
        system_parts.append(top_level_system)
    system_parts.extend(in_array_system_texts)
    if system_parts:
        oai_messages.insert(
            0,
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            },
        )

    payload: dict = {"model": model, "messages": oai_messages}

    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_tokens"] = max_tokens

    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            payload[k] = body[k]

    stop = body.get("stop_sequences")
    if stop:
        payload["stop"] = stop

    if body.get("tools"):
        payload["tools"] = _anthropic_tools_to_oai(body["tools"])
        tc = body.get("tool_choice")
        tc_oai = _anthropic_tool_choice_to_oai(tc)
        if tc_oai is not None:
            payload["tool_choice"] = tc_oai

    if body.get("stream"):
        payload["stream"] = True

    # --- Reasoning / thinking translation for OAI-compatible providers ---
    # OAI providers understand ``reasoning_effort`` but NOT Anthropic's
    # ``thinking`` dict, so we translate rather than forward verbatim.
    if reasoning_override_enabled:
        # Dashboard override is active - inject the configured level.
        payload["reasoning_effort"] = reasoning_override_level
    else:
        # Check if the caller sent Anthropic-style ``thinking`` with
        # ``type: "enabled"`` and translate the budget into an effort level.
        caller_thinking = body.get("thinking")
        if (
            isinstance(caller_thinking, dict)
            and caller_thinking.get("type") == "enabled"
        ):
            # Anthropic budget_tokens: <=4K→low, <=16K→medium, >16K→high
            budget = caller_thinking.get("budget_tokens")
            if budget is not None:
                if budget <= 4096:
                    payload["reasoning_effort"] = "low"
                elif budget <= 16384:
                    payload["reasoning_effort"] = "medium"
                else:
                    payload["reasoning_effort"] = "high"

    return payload


# ---------------------------------------------------------------------------
# Provider-specific payload fixup for the structured (OAI-compatible) path
# ---------------------------------------------------------------------------

# Fields that no OAI-compatible provider supports (Anthropic-style dicts).
_STRUCTURED_ALWAYS_STRIP = {"thinking", "reasoning"}

# Extra fields that specific providers reject on top of the always-strip set.
# NOTE: ``provider`` is intentionally only stripped for Fireworks/NVIDIA, not
# universally, because OpenRouter uses it for routing preferences.
_STRUCTURED_PROVIDER_STRIP: dict[str, set[str]] = {
    "Ollama": {"top_k", "reasoning_effort"},
    "Fireworks": {"top_k", "provider"},
    "NVIDIA NIM": {"top_k", "provider"},
    "Antigravity": {"reasoning_effort", "provider", "stop"},
}


def _structured_payload_fixup(payload: dict, provider_name: str) -> None:
    """Strip fields that the target provider does not support.

    Called in-place on *send_payload* right before the HTTP call in both
    ``_call_oai_compatible_direct`` and ``_stream_oai_compatible``.

    Every OAI-compatible provider rejects Anthropic-style ``thinking`` and
    ``reasoning`` dicts, so those are always removed.  Provider-specific
    extras (``top_k``, ``provider``) are stripped according to
    ``_STRUCTURED_PROVIDER_STRIP``.
    """
    # Capture Anthropic-style thinking/reasoning before stripping so we can
    # conditionally re-inject for providers that support a "disabled" toggle.
    orig_thinking = payload.get("thinking")
    orig_reasoning = payload.get("reasoning")

    to_strip = _STRUCTURED_ALWAYS_STRIP | _STRUCTURED_PROVIDER_STRIP.get(
        provider_name, set()
    )
    for key in to_strip:
        payload.pop(key, None)

    # Moonshot (via OpenRouter): sending reasoning_effort enables thinking mode
    # on the upstream, which then requires reasoning_content in assistant tool
    # call messages. Since we strip reasoning_content (it's not in OAI spec),
    # Moonshot rejects the request. Strip reasoning_effort for these providers.
    _STRICT_THINKING_PROVIDERS = {"Moonshot AI"}
    if provider_name in _STRICT_THINKING_PROVIDERS:
        payload.pop("reasoning_effort", None)

    # Fireworks: re-inject thinking=disabled when the caller explicitly
    # requested it and no reasoning_effort is present to take precedence.
    if provider_name == "Fireworks" and "reasoning_effort" not in payload:
        if isinstance(orig_thinking, dict) and orig_thinking.get("type") == "disabled":
            payload["thinking"] = {"type": "disabled"}
        elif isinstance(orig_reasoning, dict) and not orig_reasoning.get(
            "enabled", True
        ):
            payload["thinking"] = {"type": "disabled"}


async def _call_oai_compatible_direct(
    resolved: dict, payload: dict, request_id: str
) -> dict:
    """POST a structured OpenAI payload to a resolved provider and return the raw JSON.

    Single-shot - no retry, no timeout routing.  Raises HTTPException on any
    non-200 response and on network errors.
    """
    session: aiohttp.ClientSession = resolved["session"]
    endpoint: str = resolved["endpoint_url"]
    headers: dict = resolved["headers"]
    provider = resolved["provider_name"]

    send_payload = dict(payload)
    send_payload["model"] = resolved["api_model"]
    send_payload.pop("stream", None)
    _structured_payload_fixup(send_payload, provider)

    logger.info(
        f"[{request_id}] -> {provider} structured direct "
        f"({send_payload['model']}, {len(send_payload.get('messages', []))} msgs)"
    )
    try:
        async with session.post(endpoint, json=send_payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(
                    f"[{request_id}] {provider} {resp.status}: {error_body[:300]}"
                )
                raise HTTPException(status_code=resp.status, detail=error_body[:500])
            return await resp.json()
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        err_msg = str(e) or type(e).__name__
        logger.error(f"[{request_id}] {provider} structured direct error: {err_msg}")
        raise HTTPException(status_code=502, detail=f"{provider} error: {err_msg}")
    finally:
        if resolved.get("owns_session"):
            try:
                await session.close()
            except Exception:
                pass


async def _stream_oai_compatible(resolved: dict, payload: dict, request_id: str):
    """POST a structured OpenAI payload with stream=True, yielding parsed dict chunks.

    Decoupling the SSE framing from the translator keeps
    ``_anthropic_stream_from_openai`` backward-compatible with the legacy
    string-based upstreams while letting this generator hand out already-parsed
    dicts.  Yields each OpenAI chat.completion.chunk as a dict.  Terminates on
    ``data: [DONE]``.
    """
    session: aiohttp.ClientSession = resolved["session"]
    endpoint: str = resolved["endpoint_url"]
    headers: dict = resolved["headers"]
    provider = resolved["provider_name"]

    send_payload = dict(payload)
    send_payload["model"] = resolved["api_model"]
    send_payload["stream"] = True
    _structured_payload_fixup(send_payload, provider)

    logger.info(
        f"[{request_id}] -> {provider} structured stream "
        f"({send_payload['model']}, {len(send_payload.get('messages', []))} msgs)"
    )
    try:
        async with session.post(endpoint, json=send_payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(
                    f"[{request_id}] {provider} {resp.status}: {error_body[:300]}"
                )
                raise HTTPException(status_code=resp.status, detail=error_body[:500])

            async for frame in iter_sse_frames(resp.content.iter_any()):
                data_str = parse_sse_frame(frame).data
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    return
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        err_msg = str(e) or type(e).__name__
        logger.error(f"[{request_id}] {provider} stream error: {err_msg}")
        raise HTTPException(
            status_code=502, detail=f"{provider} stream error: {err_msg}"
        )
    finally:
        if resolved.get("owns_session"):
            try:
                await session.close()
            except Exception:
                pass


def _approx_tokens(text: str) -> int:
    """Cheap token estimate used for usage reporting (chars / 4)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _normalize_tool_use_id(raw_id: Optional[str]) -> str:
    """Return a tool_use id, generating a fresh toolu_ prefixed one when missing.

    Upstream OpenAI tool_call ids (e.g. ``call_abc123``) are preserved verbatim so
    round-trips (tool_use id → client tool_result tool_use_id → next turn's
    tool_call.id) stay consistent.  Only genuinely missing ids get a new toolu_
    identifier.
    """
    if raw_id:
        return raw_id
    return f"toolu_{uuid.uuid4().hex[:16]}"


def _openai_tool_calls_to_anthropic_blocks(tool_calls) -> list[dict]:
    """Translate an OpenAI message.tool_calls array into Anthropic tool_use blocks."""
    if not tool_calls or not isinstance(tool_calls, list):
        return []
    blocks: list[dict] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args_raw = fn.get("arguments")
        # arguments may be a JSON string (OpenAI) or already a dict (defensive)
        if isinstance(args_raw, str):
            args_raw = args_raw.strip()
            if not args_raw:
                tool_input: dict = {}
            else:
                try:
                    parsed = json.loads(args_raw)
                except json.JSONDecodeError:
                    parsed = {"_raw_arguments": args_raw}
                tool_input = parsed if isinstance(parsed, dict) else {"value": parsed}
        elif isinstance(args_raw, dict):
            tool_input = args_raw
        elif args_raw is None:
            tool_input = {}
        else:
            tool_input = {"value": args_raw}
        blocks.append(
            {
                "type": "tool_use",
                "id": _normalize_tool_use_id(tc.get("id")),
                "name": name,
                "input": tool_input,
            }
        )
    return blocks


def _openai_completion_to_anthropic_message(oai_resp: dict, request_model: str) -> dict:
    """Translate a non-streaming OpenAI chat.completion dict to an Anthropic message.

    Emits tool_use content blocks whenever ``choices[0].message.tool_calls`` is
    populated.  When no tool_calls are present the output is still a single
    text block (possibly empty) so the text-only path's consumers keep working.
    """
    choice = (oai_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    finish_reason = choice.get("finish_reason") or "stop"
    tool_calls = message.get("tool_calls") or []

    content_blocks: list[dict] = []
    if tool_calls:
        # When tool_calls are present, only include a text block if there is
        # actual text content - an empty text block would be noise.
        if isinstance(text, str) and text:
            content_blocks.append({"type": "text", "text": text})
        content_blocks.extend(_openai_tool_calls_to_anthropic_blocks(tool_calls))
        # Ensure finish_reason drives stop_reason to tool_use even if upstream
        # forgot to set it - presence of tool_calls is authoritative.
        if finish_reason not in ("tool_calls", "function_call"):
            finish_reason = "tool_calls"
    else:
        # Preserve legacy text-only behaviour: always emit one text block,
        # even when content is empty, so existing /v1/messages consumers
        # keep seeing a single-block content array.
        content_blocks.append({"type": "text", "text": text})

    stop_reason = _ANTHROPIC_STOP_REASON_MAP.get(finish_reason, "end_turn")

    usage = oai_resp.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = _approx_tokens(text)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": request_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        },
    }


async def _anthropic_stream_from_openai(
    openai_iter,
    request_model: str,
    prompt_tokens_hint: int,
):
    """Wrap an OpenAI SSE async iterator and emit Anthropic Messages SSE events.

    Accepts two chunk shapes:
      - **strings** - SSE-formatted lines ending with ``data: [DONE]\\n\\n`` (the
        legacy text-only path used by non-tool /v1/messages requests)
      - **dicts** - already-parsed OpenAI chat.completion.chunk objects (the
        structured tool-use path fed by ``_stream_oai_compatible``)

    Emits Anthropic message_start / content_block_* / message_delta / message_stop.
    Tool-call deltas open new ``tool_use`` content blocks, with ``input_json_delta``
    chunks carrying partial JSON for the ``function.arguments`` field.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    output_chars = 0
    finish_reason: Optional[str] = None

    # Content block state.  Index 0 is always pre-opened as a text block so
    # the legacy text-only flow works without extra bookkeeping - if the stream
    # then emits tool_calls we close index 0 before opening tool_use blocks.
    text_block_index = 0
    next_block_index = 1
    text_block_open = True  # index 0 opened in the header below
    # tool_state[openai_delta_index] = {
    #     "block_index": int,
    #     "id": str,
    #     "name": str,
    #     "started": bool,  # content_block_start already emitted
    #     "closed": bool,
    # }
    tool_state: dict[int, dict] = {}

    # Opening events.
    message_start_payload = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": request_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": max(0, int(prompt_tokens_hint)),
                "output_tokens": 0,
            },
        },
    }
    yield _format_anthropic_sse("message_start", message_start_payload)
    yield _format_anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": text_block_index,
            "content_block": {"type": "text", "text": ""},
        },
    )

    def _close_text_block_events() -> list[str]:
        """Return events to close the initial text block when a tool_call arrives."""
        nonlocal text_block_open
        if not text_block_open:
            return []
        text_block_open = False
        return [
            _format_anthropic_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_block_index},
            )
        ]

    def _close_tool_block_events(state: dict) -> list[str]:
        if state.get("closed") or not state.get("started"):
            state["closed"] = True
            return []
        state["closed"] = True
        return [
            _format_anthropic_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": state["block_index"]},
            )
        ]

    def _process_payload(payload: dict):
        """Yield Anthropic SSE events for one parsed OpenAI chunk dict."""
        nonlocal output_chars, finish_reason, next_block_index, text_block_open

        # Error-envelope chunks (some providers send these without a choices array)
        if isinstance(payload.get("error"), (str, dict)) and not payload.get("choices"):
            err_payload = payload["error"]
            err_text = (
                err_payload
                if isinstance(err_payload, str)
                else err_payload.get("message") or json.dumps(err_payload)
            )
            err_text = f"[upstream error] {err_text}"
            output_chars += len(err_text)
            yield _format_anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": err_text},
                },
            )
            return

        choices = payload.get("choices") or []
        if not choices:
            return
        choice0 = choices[0]
        delta = choice0.get("delta") or {}

        # Text content delta - route to the open text block at index 0.
        delta_text = delta.get("content")
        if isinstance(delta_text, str) and delta_text:
            # If a tool block is open and we receive text again, we keep the
            # text going into the original text block at index 0 - but only
            # if it's still open.  If the upstream had already emitted
            # tool_calls and closed index 0, we skip extra text (rare).
            if text_block_open:
                output_chars += len(delta_text)
                yield _format_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": delta_text},
                    },
                )

        # Tool-call deltas - open/continue tool_use blocks
        tool_call_deltas = delta.get("tool_calls") or []
        for tcd in tool_call_deltas:
            if not isinstance(tcd, dict):
                continue
            idx = tcd.get("index")
            if idx is None:
                idx = 0
            fn = tcd.get("function") or {}
            incoming_id = tcd.get("id")
            incoming_name = fn.get("name")
            incoming_args = fn.get("arguments") or ""

            state = tool_state.get(idx)
            if state is None:
                # First time we see this tool_call index.  Close the initial
                # text block (if still open) and close any previous tool block
                # so Anthropic content blocks don't overlap.
                for evt in _close_text_block_events():
                    yield evt
                for prev in tool_state.values():
                    for evt in _close_tool_block_events(prev):
                        yield evt

                state = {
                    "block_index": next_block_index,
                    "id": incoming_id or "",
                    "name": incoming_name or "",
                    "started": False,
                    "closed": False,
                    # Buffers args that arrive BEFORE the function name (some
                    # upstreams emit {arguments: "..."} on the same delta as
                    # or before {function: {name: "..."}}).  These bytes are
                    # flushed immediately after content_block_start so no
                    # partial_json is silently dropped.
                    "pending_args": "",
                }
                tool_state[idx] = state
                next_block_index += 1

            # Learn id/name lazily as upstream reveals them
            if incoming_id and not state["id"]:
                state["id"] = incoming_id
            if incoming_name and not state["name"]:
                state["name"] = incoming_name

            if not state["started"] and state["name"]:
                state["started"] = True
                yield _format_anthropic_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state["block_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": _normalize_tool_use_id(state["id"]),
                            "name": state["name"],
                            "input": {},
                        },
                    },
                )
                # Flush any args that arrived before the name was revealed.
                if state["pending_args"]:
                    yield _format_anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": state["pending_args"],
                            },
                        },
                    )
                    state["pending_args"] = ""

            if incoming_args:
                if state["started"]:
                    yield _format_anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": incoming_args,
                            },
                        },
                    )
                else:
                    # Block not yet started (no name) - buffer for flush.
                    state["pending_args"] += incoming_args

        fr = choice0.get("finish_reason")
        if fr and not finish_reason:
            finish_reason = fr

    sse_decoder = IncrementalSSEDecoder()
    upstream_error: Optional[str] = None
    try:
        async for raw_chunk in openai_iter:
            # Dict chunks (structured path) - dispatch directly.
            if isinstance(raw_chunk, dict):
                for evt in _process_payload(raw_chunk):
                    yield evt
                continue
            if not isinstance(raw_chunk, (bytes, str)):
                continue

            for frame in sse_decoder.feed(raw_chunk):
                payload_str = parse_sse_frame(frame).data
                if not payload_str or payload_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                for evt in _process_payload(payload):
                    yield evt

        for frame in sse_decoder.finalize():
            payload_str = parse_sse_frame(frame).data
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            for evt in _process_payload(payload):
                yield evt
    except (
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        aiohttp.ClientError,
        asyncio.CancelledError,
    ):
        # Client-side disconnects and upstream aborts - re-raise after the
        # finally below flushes closing events so the client (if still
        # connected) can observe a clean message_stop.
        logger.debug("Upstream/downstream disconnect in /v1/messages stream")
        upstream_error = "client_disconnect"
    except Exception as exc:  # noqa: BLE001 - defensive: never leave a stream half-open
        logger.exception("Unexpected error in _anthropic_stream_from_openai")
        upstream_error = str(exc) or type(exc).__name__
        # Surface the failure in the message body so the client gets a
        # reason rather than a silently truncated response.
        try:
            yield _format_anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {
                        "type": "text_delta",
                        "text": f"[stream error] {upstream_error}",
                    },
                },
            )
        except Exception:
            pass
    finally:
        # Close the initial text block if it's still open
        if text_block_open:
            try:
                yield _format_anthropic_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": text_block_index},
                )
            except Exception:
                pass
            text_block_open = False
        # Close any open tool_use blocks
        for state in tool_state.values():
            if state.get("started") and not state.get("closed"):
                try:
                    yield _format_anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": state["block_index"]},
                    )
                except Exception:
                    pass
                state["closed"] = True

        if upstream_error and not finish_reason:
            finish_reason = "stop"
        # Presence of any tool_use blocks forces stop_reason=tool_use even if
        # upstream forgot to set finish_reason=tool_calls.
        if tool_state and (not finish_reason or finish_reason == "stop"):
            finish_reason = "tool_calls"
        stop_reason = _ANTHROPIC_STOP_REASON_MAP.get(
            finish_reason or "stop", "end_turn"
        )
        try:
            yield _format_anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": max(1, output_chars // 4)},
                },
            )
            yield _format_anthropic_sse("message_stop", {"type": "message_stop"})
        except Exception:
            pass


async def _anthropic_messages_structured(
    body: dict,
    resolved: dict,
    request_model: str,
    is_stream: bool,
):
    """Execute the structured tool_use path for an already-resolved provider.

    Builds the OpenAI payload via :func:`_anthropic_to_oai_structured`, then
    either calls :func:`_call_oai_compatible_direct` (non-streaming) or streams
    dict chunks from :func:`_stream_oai_compatible` through the shared
    :func:`_anthropic_stream_from_openai` translator.  Request-level stats are
    recorded in the same shape as the legacy /v1/chat/completions pipeline.
    """
    request_id = uuid.uuid4().hex[:8]
    payload = _anthropic_to_oai_structured(body)

    # Rough prompt-token estimate (same chars/4 heuristic as the legacy path).
    prompt_chars = 0
    for m in payload.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            prompt_chars += len(content)
    prompt_tokens_hint = prompt_chars // 4

    if is_stream:
        payload["stream"] = True

        async def _tracked_stream():
            start = time.time()
            had_error = True
            try:
                if resolved.get("format") == "gemini":
                    chunks = stream_antigravity_structured(
                        resolved, payload, request_id
                    )
                else:
                    chunks = _stream_oai_compatible(resolved, payload, request_id)
                async for evt in _anthropic_stream_from_openai(
                    chunks, request_model, prompt_tokens_hint
                ):
                    yield evt
                had_error = False
            except (
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                aiohttp.ClientOSError,
                aiohttp.ServerDisconnectedError,
            ):
                logger.debug("Client disconnected mid-/v1/messages structured stream")
            finally:
                request_stats.record(
                    request_model, "streaming", time.time() - start, not had_error
                )

        return StreamingResponse(
            _tracked_stream(),
            media_type="text/event-stream",
            headers=_STREAMING_HEADERS,
        )

    payload.pop("stream", None)
    start = time.time()
    try:
        if resolved.get("format") == "gemini":
            data = await call_antigravity_structured(resolved, payload, request_id)
        else:
            data = await _call_oai_compatible_direct(resolved, payload, request_id)
        result = _openai_completion_to_anthropic_message(data, request_model)
        request_stats.record(request_model, "direct", time.time() - start, True)
        return result
    except Exception:
        request_stats.record(request_model, "direct", time.time() - start, False)
        raise


# --- Antigravity OAuth Login ---

# Store PKCE verifiers temporarily during OAuth flow
_oauth_states: dict[str, dict] = {}
_oauth_callback_server = None


def _cleanup_expired_oauth_states():
    """Remove OAuth state entries older than 10 minutes."""
    now = time.time()
    expired = [
        k for k, v in _oauth_states.items() if now - v.get("created_at", 0) > 600
    ]
    for k in expired:
        _oauth_states.pop(k, None)


async def start_oauth_callback_server():
    """Start a temporary server on port 51121 to handle OAuth callbacks."""
    global _oauth_callback_server

    async def callback_handler(request):
        global antigravity_auth

        if antigravity_auth.native_keychain:
            return web.Response(status=409, text="Antigravity uses the agy Keychain session. Sign in with agy.")
        _cleanup_expired_oauth_states()

        code = request.query.get("code", "")
        state = request.query.get("state", "")
        error = request.query.get("error", "")

        if error:
            safe_error = html.escape(error)
            html_content = f"""<!DOCTYPE html>
<html><head><title>OAuth Error</title><style>
body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
h1 {{ color:#f87171 }}</style></head>
<body><h1>❌ Authentication Failed</h1><p>Error: {safe_error}</p>
<a href="http://localhost:{PROXY_PORT}/config/antigravity-login">Try again</a></body></html>"""
            return web.Response(text=html_content, content_type="text/html")

        if not code or not state:
            return web.Response(
                text="<h1>Missing code or state</h1>",
                status=400,
                content_type="text/html",
            )

        stored = _oauth_states.pop(state, None)
        if not stored:
            return web.Response(
                text="<h1>Invalid state</h1>", status=400, content_type="text/html"
            )

        verifier = stored["verifier"]
        project_id = stored.get("project_id", "")

        oauth_client = get_antigravity_oauth_client()
        if oauth_client is None:
            return web.Response(
                text=f"<h1>{html.escape(antigravity_oauth_unresolved_error())}</h1>",
                status=503,
                content_type="text/html",
            )
        try:
            # Exchange code for tokens
            async with aiohttp.ClientSession() as session:
                token_payload = {
                    "client_id": oauth_client[0],
                    "client_secret": oauth_client[1],
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                    "code_verifier": verifier,
                }
                async with session.post(
                    "https://oauth2.googleapis.com/token",
                    data=token_payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        try:
                            error_code = json.loads(error_text).get("error")
                        except Exception:
                            error_code = None
                        logger.error(
                            f"Antigravity token exchange failed: HTTP {resp.status} error={error_code}"
                        )
                        html_content = (
                            f"<h1>Token exchange failed: {html.escape(error_text)}</h1>"
                        )
                        return web.Response(
                            text=html_content, status=400, content_type="text/html"
                        )

                    token_data = await resp.json()

                # Get user info
                async with session.get(
                    "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                    headers={"Authorization": f"Bearer {token_data['access_token']}"},
                ) as resp:
                    user_info = await resp.json() if resp.status == 200 else {}

                # Get project ID if not provided
                if not project_id:
                    headers = get_antigravity_headers()
                    headers["Authorization"] = f"Bearer {token_data['access_token']}"
                    async with session.post(
                        f"{ANTIGRAVITY_ENDPOINT_PROD}/v1internal:loadCodeAssist",
                        headers=headers,
                        json={
                            "metadata": {
                                "ideType": "ANTIGRAVITY",
                                "platform": "WINDOWS"
                                if sys.platform == "win32"
                                else "MACOS",
                                "pluginType": "GEMINI",
                            }
                        },
                    ) as resp:
                        if resp.status == 200:
                            load_data = await resp.json()
                            if load_data.get("cloudaicompanionProject"):
                                proj = load_data["cloudaicompanionProject"]
                                project_id = (
                                    proj.get("id", proj)
                                    if isinstance(proj, dict)
                                    else proj
                                )

            # Create new account and add to pool
            new_account = AntigravityAccount()
            new_account.access_token = token_data.get("access_token")
            new_account.refresh_token = token_data.get("refresh_token")
            new_account.email = user_info.get("email")
            new_account.project_id = project_id or ANTIGRAVITY_DEFAULT_PROJECT_ID
            new_account.expires_at = datetime.now() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            new_account.session = create_session()

            # Add to the account pool
            antigravity_auth.add_account(new_account)

            # Save to file
            _save_antigravity_auth()

            logger.info(f"Antigravity auth successful for {new_account.email}")

            # Get account count for display
            account_count = len(antigravity_auth.get_all_accounts_info())
            accounts_text = f"({account_count} account{'s' if account_count != 1 else ''} configured)"

            # Return success page with redirect
            html_content = f"""<!DOCTYPE html>
<html><head><title>Login Success</title>
<style>
body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
h1 {{ color:#4ade80 }}
.info {{ background:#1e293b; padding:16px; border-radius:8px; margin:20px 0; text-align:left }}
.label {{ color:#94a3b8; font-size:0.85rem }}
.value {{ color:#f1f5f9; font-family:monospace }}
.accounts {{ color:#67e8f9; font-size:0.9rem; margin-top:8px }}
</style></head>
<body>
<h1>✅ Authentication Successful</h1>
<div class="info">
<div style="margin-bottom:8px"><span class="label">Email:</span> <span class="value">{html.escape(new_account.email or "")}</span></div>
<div style="margin-bottom:8px"><span class="label">Project ID:</span> <span class="value">{html.escape(str(new_account.project_id or ""))}</span></div>
</div>
<div class="accounts">{accounts_text}</div>
<p>Redirecting to dashboard in 3 seconds...</p>
<script>setTimeout(function() {{ window.location.href = 'http://localhost:{PROXY_PORT}/'; }}, 3000);</script>
</body></html>"""
            return web.Response(text=html_content, content_type="text/html")

        except Exception as e:
            logger.error(f"Antigravity OAuth error: {e}")
            return web.Response(
                text=f"<h1>Error: {html.escape(str(e))}</h1>",
                status=500,
                content_type="text/html",
            )

    app = web.Application()
    app.router.add_get("/oauth-callback", callback_handler)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ANTIGRAVITY_OAUTH_PORT)
    await site.start()
    _oauth_callback_server = runner
    logger.info(f"OAuth callback server started on port {ANTIGRAVITY_OAUTH_PORT}")
    return runner


async def _fetch_claude_models() -> list[dict]:
    """Fetch available models from Anthropic API using captured auth headers."""
    if not auth.is_ready:
        return []
    session = auth.session or create_session()
    owns_session = session is not auth.session
    headers = {k: v for k, v in auth.headers.items() if k.lower() != "content-type"}
    try:
        async with session.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    {"id": m["id"], "object": "model", "owned_by": "anthropic"}
                    for m in data.get("data", [])
                    if m.get("id")
                ]
            logger.debug(f"Anthropic /v1/models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Claude model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


async def _fetch_codex_models() -> list[dict]:
    """Fetch available models from OpenAI API using captured Codex OAuth token."""
    if not codex_auth.is_ready:
        return []
    await codex_auth.refresh_if_needed()
    if not codex_auth.is_ready:
        return []
    session = codex_auth.session or create_session()
    owns_session = session is not codex_auth.session
    headers = codex_auth.get_auth_headers()
    try:
        async with session.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    {"id": m["id"], "object": "model", "owned_by": "openai"}
                    for m in data.get("data", [])
                    if m.get("id")
                ]
            logger.debug(f"OpenAI /v1/models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Codex model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


async def _fetch_zai_models() -> list[dict]:
    """Fetch available models from Z.AI API."""
    if not zai_api_key:
        return []
    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    headers = {"Authorization": f"Bearer {zai_api_key}"}
    try:
        async with session.get(
            f"{ZAI_BASE_URL}/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("data", [])
                if not models and isinstance(data, list):
                    models = data
                return [
                    {"id": f"zai:{m['id']}", "object": "model", "owned_by": "zhipuai"}
                    for m in models
                    if isinstance(m, dict) and m.get("id")
                ]
            logger.debug(f"Z.AI /models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Z.AI model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


async def _fetch_nvidia_models() -> list[dict]:
    """Fetch available models from NVIDIA NIM API."""
    if not nvidia_api_key:
        return []
    auth_session = getattr(auth, "session", None)
    session = third_party_session or auth_session or create_session()
    owns_session = session is not third_party_session and session is not auth_session
    headers = {"Authorization": f"Bearer {nvidia_api_key}"}
    try:
        async with session.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("data", [])
                if not models and isinstance(data, list):
                    models = data
                return [
                    {"id": f"nvidia:{m['id']}", "object": "model", "owned_by": "nvidia"}
                    for m in models
                    if isinstance(m, dict) and m.get("id")
                ]
            logger.debug(f"NVIDIA NIM /models returned {resp.status}")
    except Exception as e:
        logger.debug(f"NVIDIA NIM model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


async def _fetch_bytedance_models() -> list[dict]:
    """Fetch available models from the ByteDance ModelArk API.""" ""
    if not bytedance_api_key:
        return []
    auth_session = getattr(auth, "session", None)
    session = third_party_session or auth_session or create_session()
    owns_session = session is not third_party_session and session is not auth_session
    headers = {"Authorization": f"Bearer {bytedance_api_key}"}
    try:
        async with session.get(
            "https://ark.ap-southeast.bytepluses.com/api/coding/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("data", [])
                if not models and isinstance(data, list):
                    models = data
                return [
                    {
                        "id": f"bytedance:{m['id']}",
                        "object": "model",
                        "owned_by": "bytedance",
                    }
                    for m in models
                    if isinstance(m, dict) and m.get("id")
                ]
            logger.debug(f"ByteDance /models returned {resp.status}")
    except Exception as e:
        logger.debug(f"ByteDance model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


# _setup_tailscale_serve has been extracted to proxy_internal/tailscale.py
# (Phase 6) and re-exported via the facade imports at the top of this file.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Codex Proxy - multi-provider LLM gateway"
    )
    parser.add_argument(
        "--plugin-version",
        default=None,
        help="Plugin build version reported by the SkyrimNet MultiProxy launcher "
        "(logged at startup; otherwise ignored).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000, matches SkyrimNetMultiProxy.ini).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1 loopback).",
    )
    parser.add_argument(
        "--tailscale",
        action="store_true",
        help="Set up Tailscale HTTPS reverse proxy (requires Tailscale with HTTPS enabled)",
    )
    args = parser.parse_args()

    # __main__ block runs at module scope, so this assigns the module global.
    PROXY_PORT = args.port
    if args.plugin_version:
        logger.info(
            f"Launched by SkyrimNet MultiProxy plugin version {args.plugin_version}"
        )
    if args.tailscale:
        _setup_tailscale_serve([args.port])
    uvicorn.run(app, host=args.host, port=args.port, http="httptools", access_log=False)
