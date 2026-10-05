"""Config endpoints for setting provider API keys."""

import logging

from fastapi import APIRouter, Request

from proxy_internal.config_store import config_lock
from proxy_internal.endpoints._security import (
    optional_string_field,
    read_json_object,
)

router = APIRouter()
logger = logging.getLogger("proxy")


async def _update_single_key_setting(request: Request, config_key: str) -> str | None:
    """Validate a `{key: ...}` body and persist a single config key atomically."""
    import proxy

    data = await read_json_object(request)
    key = optional_string_field(data, "key")
    async with config_lock():
        cfg = proxy._load_config()
        if key:
            cfg[config_key] = key
        else:
            cfg.pop(config_key, None)
        proxy._save_config(cfg)
    return key


@router.post("/config/openrouter-key")
async def set_openrouter_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "openrouter_api_key")
    if not key:
        proxy.openrouter_api_key = None
        logger.info("OpenRouter API key cleared")
        return {"status": "cleared"}
    proxy.openrouter_api_key = key
    logger.info("OpenRouter API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/ollama-key")
async def set_ollama_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "ollama_api_key")
    if not key:
        proxy.ollama_api_key = None
        logger.info("Ollama API key cleared — using local endpoint (localhost:11434)")
        return {"status": "cleared"}
    proxy.ollama_api_key = key
    logger.info("Ollama Cloud API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/kilo-key")
async def set_kilo_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "kilo_api_key")
    if not key:
        proxy.kilo_api_key = None
        logger.info("Kilo Gateway API key cleared")
        return {"status": "cleared"}
    proxy.kilo_api_key = key
    logger.info("Kilo Gateway API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/fireworks-key")
async def set_fireworks_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "fireworks_api_key")
    if not key:
        proxy.fireworks_api_key = None
        logger.info("Fireworks API key cleared")
        return {"status": "cleared"}
    proxy.fireworks_api_key = key
    logger.info("Fireworks API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/nvidia-key")
async def set_nvidia_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "nvidia_api_key")
    if not key:
        proxy.nvidia_api_key = None
        logger.info("NVIDIA NIM API key cleared")
        return {"status": "cleared"}
    proxy.nvidia_api_key = key
    logger.info("NVIDIA NIM API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/bytedance-key")
async def set_bytedance_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "bytedance_api_key")
    if not key:
        proxy.bytedance_api_key = None
        logger.info("ByteDance API key cleared")
        return {"status": "cleared"}
    proxy.bytedance_api_key = key
    logger.info("ByteDance API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/zai-key")
async def set_zai_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "zai_api_key")
    if not key:
        proxy.zai_api_key = None
        logger.info("Z.AI API key cleared")
        return {"status": "cleared"}
    proxy.zai_api_key = key
    logger.info("Z.AI API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/xiaomi-key")
async def set_xiaomi_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "xiaomi_api_key")
    if not key:
        proxy.xiaomi_api_key = None
        logger.info("Xiaomi API key cleared")
        return {"status": "cleared"}
    proxy.xiaomi_api_key = key
    logger.info("Xiaomi API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/opencode-key")
async def set_opencode_key(request: Request):
    import proxy

    data = await read_json_object(request)
    key = optional_string_field(data, "key")
    async with config_lock():
        cfg = proxy._load_config()
        if not key:
            cfg.pop("opencode_api_key", None)
            cfg.pop("opencode_go_api_key", None)
        else:
            cfg["opencode_api_key"] = key
        proxy._save_config(cfg)
    if not key:
        proxy.opencode_api_key = None
        proxy.opencode_go_api_key = None
        logger.info("OpenCode API key cleared (both plans)")
        return {"status": "cleared"}
    proxy.opencode_api_key = key
    proxy.opencode_go_api_key = key
    logger.info("OpenCode API key configured (shared across Zen + Go)")
    return {"status": "saved"}


@router.post("/config/opencode-go-key")
async def set_opencode_go_key(request: Request):
    import proxy

    data = await read_json_object(request)
    key = optional_string_field(data, "key")
    async with config_lock():
        cfg = proxy._load_config()
        if not key:
            cfg.pop("opencode_go_api_key", None)
        else:
            cfg["opencode_go_api_key"] = key
        proxy._save_config(cfg)
    if not key:
        proxy.opencode_go_api_key = None
        logger.info("OpenCode Go API key cleared")
        return {"status": "cleared"}
    proxy.opencode_go_api_key = key
    logger.info("OpenCode Go API key configured")
    return {"status": "saved"}


@router.post("/config/vertex-key")
async def set_vertex_key(request: Request):
    import proxy

    data = await read_json_object(request)
    key = optional_string_field(data, "key")
    async with config_lock():
        cfg = proxy._load_config()
        if not key:
            cfg.pop("vertex_api_key", None)
        else:
            cfg["vertex_api_key"] = key
        proxy._save_config(cfg)
    if not key:
        proxy.vertex_api_key = None
        logger.info("Gemini API key cleared")
        await _vertex_key_rotated()
        return {"status": "cleared"}
    proxy.vertex_api_key = key
    logger.info("Gemini API key configured and saved to config.json")
    await _vertex_key_rotated()
    return {"status": "saved"}


@router.post("/config/command-code-key")
async def set_command_code_key(request: Request):
    import proxy

    key = await _update_single_key_setting(request, "command_code_api_key")
    if isinstance(getattr(proxy, "_cfg", None), dict):
        if key:
            proxy._cfg["command_code_api_key"] = key
        else:
            proxy._cfg.pop("command_code_api_key", None)
    proxy.command_code_api_key = proxy._resolve_command_code_api_key()
    if not key:
        logger.info("Command Code API key cleared from config")
        return {"status": "cleared"}
    logger.info("Command Code API key configured and saved to config.json")
    return {"status": "saved"}


async def _vertex_key_rotated() -> None:
    """Retire pooled Vertex clients built under the previous key.

    Rotation takes effect immediately (the next request builds a fresh
    client); an old client still leased by an in-flight request is closed
    when its last lease exits.
    """
    from proxy_internal.providers.vertex import vertex_key_rotated
    try:
        await vertex_key_rotated()
    except Exception as e:
        logger.warning(f"Vertex client rotation failed: {e}")
