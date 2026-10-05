"""Runtime config POST endpoints (timeouts + retries)."""
import logging

from fastapi import APIRouter, HTTPException, Request

from proxy_internal.endpoints._security import (
    require_bool_field,
    require_float_field,
    require_int_field,
    read_json_object,
)
from proxy_internal.config_store import config_lock

router = APIRouter()
logger = logging.getLogger("proxy")


@router.post("/config/timeout-routing")
async def set_timeout_routing(request: Request):
    import proxy

    data = await read_json_object(request)
    enabled = require_bool_field(data, "enabled")
    async with config_lock():
        cfg = proxy._load_config()
        cfg["timeout_routing_enabled"] = enabled
        proxy._save_config(cfg)
    proxy.timeout_routing_enabled = enabled
    logger.info(f"Timeout routing {'enabled' if enabled else 'disabled'}")
    return {"status": "ok", "timeout_routing_enabled": enabled}


@router.post("/config/timeout-cutoff")
async def set_timeout_cutoff(request: Request):
    import proxy

    data = await read_json_object(request)
    seconds = require_float_field(data, "seconds") if "seconds" in data else 6.0
    if seconds < 1.0 or seconds > 30.0:
        raise HTTPException(status_code=400, detail="Timeout must be between 1 and 30 seconds")
    async with config_lock():
        cfg = proxy._load_config()
        cfg["timeout_cutoff_seconds"] = seconds
        proxy._save_config(cfg)
    proxy.timeout_cutoff_seconds = seconds
    logger.info(f"TTFT cutoff set to {seconds}s")
    return {"status": "ok", "timeout_cutoff_seconds": seconds}


@router.post("/config/max-total")
async def set_max_total(request: Request):
    import proxy

    data = await read_json_object(request)
    seconds = require_float_field(data, "seconds") if "seconds" in data else 9.0
    if seconds < 1.0 or seconds > 60.0:
        raise HTTPException(status_code=400, detail="Max total must be between 1 and 60 seconds")
    async with config_lock():
        cfg = proxy._load_config()
        cfg["max_total_seconds"] = seconds
        proxy._save_config(cfg)
    proxy.max_total_seconds = seconds
    logger.info(f"Max total cutoff set to {seconds}s")
    return {"status": "ok", "max_total_seconds": seconds}


_VALID_REASONING_LEVELS = ("low", "medium", "high")


@router.post("/config/reasoning-override")
async def set_reasoning_override(request: Request):
    """Toggle the dashboard reasoning override.

    When enabled, every request's ``thinking``/``reasoning_effort`` is forced
    to the configured level regardless of what the caller supplied. Useful for
    CCS-routed Claude Code sessions that otherwise arrive with reasoning off.
    """
    import proxy

    data = await read_json_object(request)
    enabled = require_bool_field(data, "enabled")
    async with config_lock():
        cfg = proxy._load_config()
        cfg["reasoning_override_enabled"] = enabled
        proxy._save_config(cfg)
    proxy.reasoning_override_enabled = enabled
    logger.info(
        f"Reasoning override {'enabled' if enabled else 'disabled'} "
        f"(level={proxy.reasoning_override_level})"
    )
    return {
        "status": "ok",
        "reasoning_override_enabled": enabled,
        "reasoning_override_level": proxy.reasoning_override_level,
    }


@router.post("/config/reasoning-level")
async def set_reasoning_level(request: Request):
    """Set the reasoning effort level used when the override is active.

    Level maps to (a) Anthropic ``thinking.budget_tokens`` and (b) OpenAI-style
    ``reasoning_effort`` — both are injected into ``extra_params`` so whichever
    field the downstream provider honours is set.
    """
    import proxy

    data = await read_json_object(request)
    level = data.get("level")
    if not isinstance(level, str) or level not in _VALID_REASONING_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"level must be one of {_VALID_REASONING_LEVELS}",
        )
    async with config_lock():
        cfg = proxy._load_config()
        cfg["reasoning_override_level"] = level
        proxy._save_config(cfg)
    proxy.reasoning_override_level = level
    logger.info(f"Reasoning override level set to {level}")
    return {
        "status": "ok",
        "reasoning_override_enabled": proxy.reasoning_override_enabled,
        "reasoning_override_level": level,
    }


@router.post("/config/reasoning-rewrite")
async def set_reasoning_rewrite(request: Request):
    """Toggle the reasoning rewrite feature.

    When enabled, if an upstream returns only reasoning tokens with no visible
    content, the proxy makes a secondary call to rewrite the reasoning into
    NPC dialogue. Uses Fireworks API credits.
    """
    import proxy

    data = await read_json_object(request)
    enabled = require_bool_field(data, "enabled")
    async with config_lock():
        cfg = proxy._load_config()
        cfg["reasoning_rewrite_enabled"] = enabled
        proxy._save_config(cfg)
    proxy.reasoning_rewrite_enabled = enabled
    logger.info(f"Reasoning rewrite {'enabled' if enabled else 'disabled'}")
    return {"status": "ok", "reasoning_rewrite_enabled": enabled}


@router.post("/config/max-retries")
async def set_max_retries(request: Request):
    """Update the retry budget. Clamps to 0..10 and persists to config.json."""
    import proxy

    data = await read_json_object(request)
    n = require_int_field(data, "max_retries", minimum=0, maximum=10)
    async with config_lock():
        cfg = proxy._load_config()
        cfg["max_retries"] = n
        proxy._save_config(cfg)
    proxy.max_retries = n
    logger.info(f"Retry policy updated: max_retries={n}")
    return {"status": "saved", "max_retries": n}
