"""Per-provider concurrency limiting with a bounded queue wait.

Subscription plans meter *simultaneous* requests, not requests per minute: z.ai's
coding plan answers 429 once more than six calls are in flight at once, however
spaced they are. This module makes the proxy wait in line for a free slot rather
than tripping that cap, and gives up with a 503 (so the caller can fail over)
once ``queue_timeout_ms`` elapses.

Limits come from ``providers.<name>.max_concurrent`` in config.json, falling back
to :data:`_DEFAULT_LIMITS`. ``0`` means unlimited. Limits may be edited live
through ``POST /config/providers``; a changed limit takes effect as soon as the
provider's pool drains (rebuilding a pool mid-flight would over-admit callers
already holding slots).

Ported from the community proxy that shipped per-provider request limits first.

Scope: slots are taken by the OpenAI-compatible chat path (``/v1/chat/
completions``, what SkyrimNet and the dashboard use). The Anthropic messages
path and the internal timeout-routing fallbacks call providers directly and are
not pooled — a slot covers one client request, so a fallback cascade cannot
deadlock against the pool it is already holding.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from fastapi import HTTPException

from proxy_internal import provider_config

logger = logging.getLogger("proxy")

# Providers with a documented plan-wide cap on simultaneous requests. z.ai's
# coding plans (Lite/Pro/Max) allow 6 concurrent calls; everything else stays
# unlimited until config.json says otherwise.
_DEFAULT_LIMITS: dict[str, int] = {"zai": 6}


class _Pool:
    """Slot bookkeeping for one provider."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.queued = 0
        self.acquired_total = 0
        self.timed_out_total = 0

    def stats(self) -> dict:
        return {
            "limit": self.limit,
            "active": self.active,
            "queued": self.queued,
            "acquired_total": self.acquired_total,
            "timed_out_total": self.timed_out_total,
        }


_pools: dict[str, _Pool] = {}


def _pool_for(provider: str, limit: int) -> _Pool:
    """Return the live pool for ``provider``, rebuilding an idle one whose limit changed."""
    pool = _pools.get(provider)
    if pool is not None and pool.limit == limit:
        return pool
    if pool is None or (pool.active == 0 and pool.queued == 0):
        # No await between the idle check and the swap, so a pool only ever
        # replaces one that nobody is holding or waiting on.
        pool = _Pool(limit)
        _pools[provider] = pool
    return pool


class Slot:
    """A held slot. ``release`` is idempotent and safe to call from a ``finally``."""

    def __init__(self, pool: Optional[_Pool]) -> None:
        self._pool = pool
        self._released = False

    def release(self) -> None:
        if self._pool is None or self._released:
            return
        self._released = True
        self._pool.semaphore.release()
        self._pool.active -= 1


def _limit_for(provider: str) -> int:
    """Configured simultaneous-request cap for ``provider`` (0 = unlimited)."""
    configured = provider_config.get_provider_setting(
        provider, "max_concurrent", _DEFAULT_LIMITS.get(provider, 0)
    )
    return max(0, configured)


async def acquire(provider: str, *, request_id: str = "") -> Slot:
    """Wait for a free slot for ``provider``, or raise HTTP 503 on timeout.

    Returns immediately with an inert slot when the provider is unlimited.
    """
    limit = _limit_for(provider)
    if limit <= 0:
        return Slot(None)

    pool = _pool_for(provider, limit)
    timeout_s = max(
        0,
        provider_config.get_provider_setting(
            provider, "queue_timeout_ms", provider_config.DEFAULTS["queue_timeout_ms"]
        ),
    ) / 1000.0

    pool.queued += 1
    started = time.monotonic()
    try:
        if timeout_s > 0:
            await asyncio.wait_for(pool.semaphore.acquire(), timeout=timeout_s)
        else:
            await pool.semaphore.acquire()
    except asyncio.TimeoutError:
        pool.timed_out_total += 1
        logger.error(
            f"[{request_id}] {provider} concurrency queue timed out after {timeout_s:.0f}s "
            f"({pool.active}/{pool.limit} active, {pool.queued} waiting)"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"{provider} is already running {pool.limit} requests at once and none freed up "
                f"within {timeout_s:.0f}s. Raise \"max_concurrent\" for {provider} in config.json "
                f"if your plan allows more."
            ),
        )
    finally:
        pool.queued -= 1

    pool.active += 1
    pool.acquired_total += 1
    waited = time.monotonic() - started
    if waited > 0.05:
        logger.info(
            f"[{request_id}] {provider} waited {waited:.1f}s for a concurrency slot "
            f"({pool.active}/{pool.limit} active)"
        )
    return Slot(pool)


async def hold_slot(events: AsyncIterator, slot: Slot) -> AsyncIterator:
    """Yield ``events``, releasing ``slot`` when the stream ends or is closed."""
    try:
        async for event in events:
            yield event
    finally:
        slot.release()


def snapshot() -> dict:
    """Per-provider pool telemetry for /health (idle pools included once known)."""
    out = {name: pool.stats() for name, pool in _pools.items()}
    known = set(_DEFAULT_LIMITS) | set(_pools) | set(provider_config.get_all_settings())
    for name in known:
        limit = _limit_for(name)
        if limit > 0 and name not in out:
            out[name] = {
                "limit": limit,
                "active": 0,
                "queued": 0,
                "acquired_total": 0,
                "timed_out_total": 0,
            }
    return out


def reset() -> None:
    """Drop every pool. Test helper — a live proxy has no reason to call this."""
    _pools.clear()
