"""Runtime accessors for the `providers` block in config.json.

Fields supported per provider:
- enabled  (bool, default True)
- timeout_ms (int, default 300000, clamped to [1000, 3600000])
- priority (int, default 1000, clamped to [0, 10000])
- max_concurrent (int, default 0 = unlimited, clamped to [0, 64])
- queue_timeout_ms (int, default 60000, clamped to [0, 600000], 0 = wait forever)

Missing values fall back to the defaults. Unknown provider names in config
are tolerated (returned as empty dict via .get), but patches to the module
via set_provider_setting() are validated against KNOWN_PROVIDER_NAMES.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PRIORITY_MIN, PRIORITY_MAX = 0, 10000
TIMEOUT_MIN_MS, TIMEOUT_MAX_MS = 1000, 3600000
MAX_CONCURRENT_MIN, MAX_CONCURRENT_MAX = 0, 64
QUEUE_TIMEOUT_MIN_MS, QUEUE_TIMEOUT_MAX_MS = 0, 600000

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "timeout_ms": 300000,
    "priority": 1000,
    "max_concurrent": 0,
    "queue_timeout_ms": 60000,
}

# Populated at first access to avoid import-time cycles with registry.py.
KNOWN_PROVIDER_NAMES: set[str] = set()

# In-memory cache of the `providers` block from config.json.
# Reload explicitly via reload_from_config().
_cache: dict[str, dict[str, Any]] = {}


def _populate_known_names() -> None:
    global KNOWN_PROVIDER_NAMES
    if KNOWN_PROVIDER_NAMES:
        return
    from proxy_internal.registry import REGISTRY
    KNOWN_PROVIDER_NAMES = {p.name for p in REGISTRY}


def reload_from_config() -> None:
    """Re-read the `providers` block from config.json into _cache.

    Unknown provider names are kept in the cache (tolerant), but a log warning
    fires once per unknown name. Malformed entries (non-dict values) are dropped.
    """
    global _cache
    from proxy_internal.config_store import _load_config
    cfg = _load_config()
    raw = cfg.get("providers", {})
    if not isinstance(raw, dict):
        logger.warning("config.json 'providers' is not a dict; ignoring")
        _cache = {}
        return
    cleaned: dict[str, dict[str, Any]] = {}
    _populate_known_names()
    for name, patch in raw.items():
        if not isinstance(patch, dict):
            logger.warning("config.json providers.%s is not a dict; ignoring", name)
            continue
        if KNOWN_PROVIDER_NAMES and name not in KNOWN_PROVIDER_NAMES:
            logger.warning("config.json providers.%s is not a known provider", name)
        try:
            cleaned[name] = validate_patch(patch)
        except ValueError as e:
            logger.warning("config.json providers.%s rejected: %s", name, e)
    _cache = cleaned


def get_provider_setting(name: str, field: str, default: Any) -> Any:
    """Return the configured value for (name, field), falling back to default."""
    entry = _cache.get(name) or {}
    if field in entry:
        return entry[field]
    return default


def validate_patch(patch: dict) -> dict:
    """Validate and clamp a partial provider-settings patch.

    - `enabled` must be bool (raises ValueError otherwise).
    - `priority`, `timeout_ms`, `max_concurrent` and `queue_timeout_ms` must be
      ints (not bool, not float); clamped.
    - Unknown fields are dropped.
    """
    out: dict[str, Any] = {}
    if "enabled" in patch:
        val = patch["enabled"]
        if not isinstance(val, bool):
            raise ValueError("enabled must be a bool")
        out["enabled"] = val
    if "priority" in patch:
        val = patch["priority"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("priority must be an int")
        out["priority"] = max(PRIORITY_MIN, min(PRIORITY_MAX, val))
    if "timeout_ms" in patch:
        val = patch["timeout_ms"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("timeout_ms must be an int")
        out["timeout_ms"] = max(TIMEOUT_MIN_MS, min(TIMEOUT_MAX_MS, val))
    if "max_concurrent" in patch:
        val = patch["max_concurrent"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("max_concurrent must be an int")
        out["max_concurrent"] = max(MAX_CONCURRENT_MIN, min(MAX_CONCURRENT_MAX, val))
    if "queue_timeout_ms" in patch:
        val = patch["queue_timeout_ms"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("queue_timeout_ms must be an int")
        out["queue_timeout_ms"] = max(
            QUEUE_TIMEOUT_MIN_MS, min(QUEUE_TIMEOUT_MAX_MS, val)
        )
    return out


def get_all_settings() -> dict[str, dict[str, Any]]:
    """Return a deep copy of all cached provider settings (for GET endpoint)."""
    return {name: dict(entry) for name, entry in _cache.items()}


def apply_patch(name: str, patch: dict) -> dict[str, Any]:
    """Validate + merge a patch for `name` without mutating the cache."""
    _populate_known_names()
    if KNOWN_PROVIDER_NAMES and name not in KNOWN_PROVIDER_NAMES:
        raise ValueError(f"unknown provider: {name}")
    cleaned = validate_patch(patch)
    merged = dict(_cache.get(name, {}))
    merged.update(cleaned)
    return merged


def commit_patch(name: str, settings: dict[str, Any]) -> None:
    """Persist the merged settings for `name` into the in-memory cache."""
    _populate_known_names()
    if KNOWN_PROVIDER_NAMES and name not in KNOWN_PROVIDER_NAMES:
        raise ValueError(f"unknown provider: {name}")
    _cache[name] = dict(settings)


def export_for_save() -> dict[str, dict[str, Any]]:
    """Return the cache in a form suitable for persistence under config['providers']."""
    return get_all_settings()
