"""GET/POST /config/providers — runtime control for the providers config block."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from proxy_internal import provider_config
from proxy_internal.config_store import config_lock

router = APIRouter()


class ProviderPatch(BaseModel):
    # Accept raw types for the numeric/bool fields so validation errors surface
    # as HTTP 400 (via validate_patch ValueError) instead of Pydantic's 422.
    name: str
    enabled: Optional[Any] = None
    timeout_ms: Optional[Any] = None
    priority: Optional[Any] = None
    max_concurrent: Optional[Any] = None
    queue_timeout_ms: Optional[Any] = None


@router.get("/config/providers")
async def get_providers():
    """Return the current `providers` block from config.json as in-memory cache."""
    provider_config.reload_from_config()
    return provider_config.get_all_settings()


@router.post("/config/providers")
async def patch_provider(body: ProviderPatch):
    """Apply a partial patch to one provider's settings.

    Validates name against REGISTRY and clamps/validates numeric fields.
    Persists to config.json atomically under the config lock.
    """
    patch = {k: v for k, v in body.model_dump().items() if k != "name" and v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="at least one field must be set")
    try:
        import proxy
        async with config_lock():
            merged = provider_config.apply_patch(body.name, patch)
            cfg = proxy._load_config()
            cfg.setdefault("providers", {})[body.name] = merged
            proxy._save_config(cfg)
            provider_config.commit_patch(body.name, merged)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": body.name, "settings": merged}
