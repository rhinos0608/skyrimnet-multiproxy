"""Group live /v1/models output into per-provider buckets.

Grouping rule (mirrors registry.detect_provider):
- For each model id, walk REGISTRY in order.
- First entry whose `detector(id)` returns True claims the id.
- Models claimed by no detector fall to the default bucket (detector=None,
  always last in REGISTRY — conventionally `claude`).

Bucket values are sorted alphabetically so listing-tool output is
deterministic across calls (upstream `/v1/models` ordering is not stable).
"""
from __future__ import annotations

from typing import Iterable


def group_models_by_provider(
    live_models: Iterable[dict],
    registry: list,
) -> dict[str, list[str]]:
    """Return {provider_name: sorted_model_ids} for providers with at least one match."""
    buckets: dict[str, list[str]] = {}
    default_name: str | None = None
    for p in registry:
        if p.detector is None:
            default_name = p.name

    for m in live_models:
        model_id = m.get("id") if isinstance(m, dict) else None
        if not model_id:
            continue
        claimed_by: str | None = None
        for p in registry:
            if p.detector is None:
                continue
            try:
                if p.detector(model_id):
                    claimed_by = p.name
                    break
            except Exception:
                # A detector raising shouldn't crash the listing tools.
                continue
        if claimed_by is None:
            claimed_by = default_name
        if claimed_by is None:
            continue  # no default configured — drop
        buckets.setdefault(claimed_by, []).append(model_id)

    for name in buckets:
        buckets[name].sort()
    return buckets
