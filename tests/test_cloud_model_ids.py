"""Cloud model-id evidence (static; no network).

Live-verified 2026-09-30 through a local proxy instance
(POST /v1/chat/completions, max_tokens=16, "Reply with exactly: ok"):

- ollama:glm-5.3            -> 402 Insufficient account funds
- ollama:kimi-k3            -> 402 Insufficient account funds
- opencode:glm-5.3          -> 402 Insufficient account funds
- opencode:deepseek-v4-flash -> 402 Insufficient account funds

Earlier evidence (parent baseline): ollama:glm-5.2 -> 402 free-usage
credits required; ollama:llama3.1 -> 404 not on Ollama Cloud;
opencode:deepseek-v4-flash-free -> 400 Model is unavailable;
opencode:big-pickle -> 401 FreeTierError; opencode:glm-5.3 -> 402.

Conclusion: the account tier — not the id spelling — blocks every
newer candidate, so both defaults stay as-is until entitlement changes.
These tests pin that decision: if someone bumps either default, they
must have fresh 200-with-content evidence, and the id must still route
to its own provider.

Antigravity evidence 2026-09-30 (same probe shape, this account):
- antigravity-gemini-3-pro / 3-pro-low -> 200 but upstream "no longer
  available, switch to 3.1 Pro" notice (FAIL: notice, not an answer)
- antigravity-gemini-3.1-pro(-high) -> 400 INVALID_ARGUMENT; -low ->
  200 with empty content (FAIL)
- antigravity-claude-sonnet-4-6, opus-4-5-thinking, gpt-oss-120b-medium
  -> 404; gemini-2.5-pro -> 503 no capacity
- antigravity-gemini-3-flash / 3.1-flash / 3.1-flash-lite -> 200 "ok"
  (PASS); registry default + /v1/models now ship the flash ids only.
"""

from proxy_internal.registry import REGISTRY, detect_provider


def _by_name(name):
    for p in REGISTRY:
        if p.name == name:
            return p
    raise AssertionError(f"provider {name!r} missing from registry")


def test_ollama_default_unchanged_pending_entitlement():
    assert _by_name("ollama").default_model == "ollama:llama3.1"


def test_opencode_zen_default_unchanged_pending_entitlement():
    assert _by_name("opencode_zen").default_model == "opencode:deepseek-v4-flash-free"


def test_cloud_defaults_still_route_to_own_provider():
    for name in ("ollama", "opencode_zen"):
        p = _by_name(name)
        assert p.detector(p.default_model), (name, p.default_model)
        assert detect_provider(p.default_model).name == name, (name, p.default_model)
