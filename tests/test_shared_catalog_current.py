"""Shared suggestion surfaces reflect current provider research (no stale IDs)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = (REPO / "proxy_internal" / "endpoints" / "models.py").read_text()
DASH = (REPO / "proxy_internal" / "dashboard.py").read_text()
DASH_EP = (REPO / "proxy_internal" / "endpoints" / "dashboard.py").read_text()

DEAD = [
    "mimo-v2-pro",
    "mimo-v2-omni",
    "mimo-v2-flash",
    "big-pickle",
    "minimax-m2.5-free",
    "nemotron-3-super-free",
    "gemma-3-27b",
    "bytedance:glm-4.7",
    '"glm-4.7"',
    "kimi-k2.6\",\n            \"opencode-go:mimo-v2.5\",\n",
    "gpt-oss-120b-medium",
    "claude-opus-4-5-thinking",
]


def test_no_dead_ids_in_shared_catalogs():
    import re

    for dead in DEAD:
        pat = re.compile(r'value="[^"]*' + re.escape(dead) + r'"|"id":\s*"[^"]*' + re.escape(dead))
        for name, text in [("models.py", MODELS), ("dashboard.py", DASH), ("dashboard-ep", DASH_EP)]:
            assert not pat.search(text), f"{name} still suggests {dead!r}"


def test_current_ids_present():
    for good in [
        "antigravity-gemini-3-flash",
        "antigravity-gemini-3.1-flash",
        "xiaomi:mimo-v2.6-flash",
        "xiaomi:mimo-v2.6-pro",
        "opencode-go:kimi-k3",
        "opencode-go:mimo-v2.6-flash",
        "fireworks:kimi-k3",
        "nvidia:kimi-k3",
    ]:
        assert good in MODELS or good in DASH, f"missing current id {good!r}"


def test_antigravity_dashboard_matches_catalog_trio():
    trio = [
        "antigravity-gemini-3-flash",
        "antigravity-gemini-3.1-flash",
        "antigravity-gemini-3.1-flash-lite",
    ]
    for mid in trio:
        assert mid in DASH_EP
        assert mid in DASH
    assert "antigravity-claude-sonnet-4-6" not in DASH_EP
    assert "antigravity-claude-sonnet-4-6" not in DASH


