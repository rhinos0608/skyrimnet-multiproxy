"""Safe read-only discovery of existing stored credentials.

Read-only helper: inspects Pi agent stores and CLI auth files for
already-saved keys. Performs no login, no network I/O, never copies
credentials anywhere, and never executes ``!shell`` command values.

Priority (caller handles explicit env vars and proxy encrypted config
first, before calling here):

``discover_command_code_key``:
    1. Pi ``auth.json`` provider ``command-code``: ``api_key`` record,
       or *synthetic* OAuth record only.
    2. Pi global ``settings.json``: ``command-code.apiKey`` or nested
       ``command-code`` mapping.
    3. CLI store ``~/.commandcode/auth.json`` top-level ``apiKey``.

``discover_opencode_key``:
    1. Pi ``auth.json`` ``opencode-go``/``opencode`` ``api_key`` record
       matching ``plan``, then the other id (Zen/Go share one key).
    2. CLI ``~/.local/share/opencode/auth.json`` entries
       ``opencode``/``opencode-go`` with ``type == 'api'``, preferred
       plan first, then the other id.

``discover_vertex_key`` / ``discover_openrouter_key`` /
``discover_ollama_key``: Pi ``google-vertex`` / ``openrouter`` /
``ollama`` ``api_key`` record respectively.

``discover_claude_oauth`` / ``discover_codex_oauth`` /
``discover_antigravity_oauth``: Pi
``anthropic`` / ``openai-codex`` / ``antigravity``
OAuth record access token, valid only while the record's ``expires``
epoch-ms timestamp is in the future. These return bearer tokens,
not API keys.

``discover_all``: redacted ``ok``/``expired``/``missing`` status per
provider id; never includes credential values.

Safety rules:
- ``!command`` values are never executed; they yield no key.
- ``$ENV`` interpolation resolves ONLY an exact ``$VARNAME`` or
  ``${VARNAME}`` reference from the process environment (no shell).
  Anything else containing ``$`` is treated as unresolved -> no key.
- Files are size-capped, must hold a JSON object, and world-writable
  files are ignored on POSIX. Malformed/missing files -> no key.
- Nothing is ever logged with credential values or filesystem paths.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger("proxy")

# Number.MAX_SAFE_INTEGER: what pi-command-code returns from its synthetic
# OAuth login (access == refresh == API key). Only this exact shape counts
# as a usable key; arbitrary real OAuth bearer tokens are NOT assumed to
# work as API keys.
_SYNTHETIC_OAUTH_EXPIRES = 9007199254740991

_MAX_JSON_BYTES = 256 * 1024

_EXACT_ENV_RE = re.compile(r"^(?:\$([A-Za-z_][A-Za-z0-9_]*)|\$\{([A-Za-z_][A-Za-z0-9_]*)\})$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clean_resolved_value(value: Any) -> Optional[str]:
    """Trim padding; reject empty, control chars, or shell expressions."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if _CONTROL_CHARS_RE.search(cleaned):
        return None
    if cleaned.startswith("!"):
        return None
    return cleaned


def _agent_dir() -> str:
    override = os.environ.get("PI_CODING_AGENT_DIR")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".pi", "agent")


def _safe_load_json_object(path: str) -> Optional[dict]:
    """Read a JSON object file defensively; None on any problem."""
    try:
        if not os.path.isfile(path):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size > _MAX_JSON_BYTES:
            logger.debug("Credential store oversized; ignoring")
            return None
        if os.name == "posix":
            try:
                mode = os.stat(path).st_mode
            except OSError:
                return None
            if mode & stat.S_IWOTH:
                logger.debug("Credential store world-writable; ignoring")
                return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        logger.debug("Credential store unreadable; ignoring")
        return None
    if not isinstance(data, dict):
        logger.debug("Credential store malformed; ignoring")
        return None
    return data


def _resolve_literal_key(raw: Any) -> Optional[str]:
    """Return usable literal key or None.

    Never executes ``!`` commands. Resolves only exact ``$VAR``/``${VAR}``
    from process env (after trimming); anything else with ``$`` is
    unresolved -> None. Trims padding; rejects empty, control chars,
    and ``!``-prefixed values after trimming.
    """
    if not isinstance(raw, str) or not raw:
        return None
    if _CONTROL_CHARS_RE.search(raw):
        return None
    text = raw.strip()
    if not text:
        return None
    if _CONTROL_CHARS_RE.search(text):
        return None
    if text.startswith("!"):
        return None
    if "$" not in text:
        return text
    match = _EXACT_ENV_RE.match(text)
    if not match:
        return None
    value = os.environ.get(match.group(1) or match.group(2))
    return _clean_resolved_value(value)


def _api_key_from_record(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    if record.get("type") != "api_key":
        return None
    return _resolve_literal_key(record.get("key"))


def _synthetic_oauth_key(record: Any) -> Optional[str]:
    """Accept ONLY synthetic OAuth (access == refresh, max-safe expiry)."""
    if not isinstance(record, dict):
        return None
    if record.get("type") != "oauth":
        return None
    access = record.get("access")
    refresh = record.get("refresh")
    expires = record.get("expires")
    if not isinstance(access, str) or not access:
        return None
    if access != refresh:
        return None
    if expires != _SYNTHETIC_OAUTH_EXPIRES:
        return None
    return _clean_resolved_value(access)


def _pi_auth_record(provider_id: str) -> Any:
    data = _safe_load_json_object(os.path.join(_agent_dir(), "auth.json"))
    if data is None:
        return None
    return data.get(provider_id)


def _pi_settings_command_code_key() -> Optional[str]:
    data = _safe_load_json_object(os.path.join(_agent_dir(), "settings.json"))
    if data is None:
        return None
    direct = data.get("command-code.apiKey")
    resolved = _resolve_literal_key(direct)
    if resolved:
        return resolved
    nested = data.get("command-code")
    if isinstance(nested, dict):
        return _resolve_literal_key(nested.get("apiKey"))
    return None


def discover_command_code_key() -> Optional[str]:
    """Return stored Command Code key, or None. Read-only."""
    record = _pi_auth_record("command-code")
    key = _api_key_from_record(record)
    if key:
        return key
    key = _synthetic_oauth_key(record)
    if key:
        return key
    key = _pi_settings_command_code_key()
    if key:
        return key
    cli = _safe_load_json_object(
        os.path.join(os.path.expanduser("~"), ".commandcode", "auth.json")
    )
    if cli is not None:
        return _resolve_literal_key(cli.get("apiKey"))
    return None


def discover_opencode_key(plan: str = "Go") -> Optional[str]:
    """Return stored OpenCode key for plan ``Go`` or ``Zen``, or None."""
    preferred = "opencode-go" if plan == "Go" else "opencode"
    fallback = "opencode" if plan == "Go" else "opencode-go"
    for provider_id in (preferred, fallback):
        key = _api_key_from_record(_pi_auth_record(provider_id))
        if key:
            return key
    cli = _safe_load_json_object(
        os.path.join(
            os.path.expanduser("~"), ".local", "share", "opencode", "auth.json"
        )
    )
    if cli is None:
        return None
    for provider_id in (preferred, fallback):
        entry = cli.get(provider_id)
        if (
            isinstance(entry, dict)
            and entry.get("type") == "api"
            and _resolve_literal_key(entry.get("key"))
        ):
            return _resolve_literal_key(entry.get("key"))
    return None


def _fresh_oauth_access(provider_id: str) -> Optional[str]:
    """Return Pi OAuth access token iff ``expires`` epoch-ms is in future."""
    import time

    try:
        record = _pi_auth_record(provider_id)
    except Exception:
        logger.debug("Credential store unreadable; ignoring")
        return None
    if not isinstance(record, dict):
        return None
    if record.get("type") != "oauth":
        return None
    token = _resolve_literal_key(record.get("access"))
    if token is None:
        return None
    expires = record.get("expires")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return None
    try:
        now_ms = int(time.time() * 1000)
    except Exception:
        logger.debug("Credential store unreadable; ignoring")
        return None
    if not expires > now_ms:
        return None
    return token


def _oauth_status(provider_id: str) -> str:
    """Redacted OAuth status: 'ok', 'expired', or 'missing'."""
    try:
        record = _pi_auth_record(provider_id)
    except Exception:
        return "missing"
    if record is None:
        return "missing"
    return "ok" if _fresh_oauth_access(provider_id) is not None else "expired"


def discover_vertex_key() -> Optional[str]:
    """Return Pi ``google-vertex`` api_key record, or None."""
    return _api_key_from_record(_pi_auth_record("google-vertex"))


def discover_openrouter_key() -> Optional[str]:
    """Return Pi ``openrouter`` api_key record, or None."""
    return _api_key_from_record(_pi_auth_record("openrouter"))


def discover_ollama_key() -> Optional[str]:
    """Return Pi ``ollama`` api_key record (Ollama Cloud key), or None."""
    return _api_key_from_record(_pi_auth_record("ollama"))


def discover_claude_oauth() -> Optional[str]:
    """Return Pi ``anthropic`` OAuth access token iff unexpired; not an API key."""
    return _fresh_oauth_access("anthropic")


def discover_codex_oauth() -> Optional[dict]:
    """Return Pi ``openai-codex`` OAuth token dict iff unexpired; not an API key.

    Returns ``{"access_token": ..., "account_id": ...}`` where
    ``account_id`` is the stored ``accountId`` (or None when absent),
    or None when the record is missing, expired, or malformed.
    """
    token = _fresh_oauth_access("openai-codex")
    if token is None:
        return None
    try:
        record = _pi_auth_record("openai-codex")
    except Exception:
        logger.debug("Credential store unreadable; ignoring")
        return None
    account_id = None
    if isinstance(record, dict):
        raw_aid = record.get("accountId", record.get("account_id"))
        if isinstance(raw_aid, str):
            account_id = _clean_resolved_value(raw_aid)
    return {"access_token": token, "account_id": account_id}


@dataclass(frozen=True)
class NativeAntigravityToken:
    """Access-only snapshot of the CLI-owned session; never persisted by proxy."""

    access_token: str = field(repr=False)
    expires_at: datetime


def native_antigravity_enabled() -> bool:
    """Use the native agy session by default on macOS."""
    return sys.platform == "darwin"


def discover_native_antigravity_token() -> Optional[NativeAntigravityToken]:
    """Read only agy's Keychain item; discard refresh tokens and identity data.

    go-keyring on Darwin stores plain JSON or a prefixed base64/hex string.
    No unlock, login, token refresh, shell execution, or Keychain writes.
    """
    if not native_antigravity_enabled():
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", "gemini",
             "-a", "antigravity", "-w"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=8,
        )
        if result.returncode or len(result.stdout) > _MAX_JSON_BYTES:
            return None
        raw = result.stdout.strip()
        if raw.startswith(b"go-keyring-base64:"):
            raw = base64.b64decode(raw[len(b"go-keyring-base64:"):], validate=True)
        elif raw.startswith(b"go-keyring-encoded:"):
            raw = bytes.fromhex(raw[len(b"go-keyring-encoded:"):].decode("ascii"))
        data = json.loads(raw)
        token = data.get("token") if isinstance(data, dict) else None
        if not isinstance(token, dict) or token.get("token_type", "Bearer").lower() != "bearer":
            return None
        access = _clean_resolved_value(token.get("access_token"))
        expiry = token.get("expiry")
        if not access or not isinstance(expiry, str):
            return None
        expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc) + timedelta(minutes=5):
            return None
        return NativeAntigravityToken(access, expires_at)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError, binascii.Error):
        # Exceptions can contain subprocess output, including the credential.
        logger.debug("Native agy session unavailable")
        return None


def discover_antigravity_oauth() -> Optional[str]:
    """Return Pi ``antigravity`` OAuth access token iff unexpired; not a key."""
    return _fresh_oauth_access("antigravity")


def discover_all() -> dict:
    """Map stable provider id -> 'ok'/'expired'/'missing'; never values."""
    status: dict = {}
    for pid in ("vertex", "openrouter", "ollama"):
        status[pid] = "missing"
    if discover_vertex_key() is not None:
        status["vertex"] = "ok"
    if discover_openrouter_key() is not None:
        status["openrouter"] = "ok"
    if discover_ollama_key() is not None:
        status["ollama"] = "ok"
    status["claude"] = _oauth_status("anthropic")
    status["codex"] = _oauth_status("openai-codex")
    status["antigravity"] = _oauth_status("antigravity")
    try:
        cc = discover_command_code_key()
    except Exception:
        cc = None
    status["command-code"] = "ok" if cc is not None else "missing"
    try:
        oc_go = discover_opencode_key("Go")
        oc_zen = discover_opencode_key("Zen")
    except Exception:
        oc_go = oc_zen = None
    status["opencode-go"] = "ok" if oc_go is not None else "missing"
    status["opencode-zen"] = "ok" if oc_zen is not None else "missing"
    return status
