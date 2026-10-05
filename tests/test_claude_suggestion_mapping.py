"""Claude stale-suggestion mapping: 4.6 / deprecated 4.5 / retired 3.x -> 5.x."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy  # noqa: E402
from proxy_internal.providers.claude import (  # noqa: E402
    _build_api_body,
    resolve_claude_model,
)


def test_stale_suggestions_remap_to_current():
    assert resolve_claude_model("claude-sonnet-4-6") == "claude-sonnet-5-5"
    assert (
        resolve_claude_model("claude-sonnet-4-5-20250929")
        == "claude-sonnet-5-5"
    )
    assert resolve_claude_model("claude-opus-4-6") == "claude-opus-5-5"


def test_retired_3x_remap_to_current():
    assert (
        resolve_claude_model("claude-3-7-sonnet-20250219")
        == "claude-sonnet-5-5"
    )
    assert (
        resolve_claude_model("claude-3-5-sonnet-20241022")
        == "claude-sonnet-5-5"
    )
    assert (
        resolve_claude_model("claude-3-opus-20240229") == "claude-opus-5-5"
    )


def test_current_ids_untouched():
    for current in (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-haiku-4-5-20251001",
    ):
        assert resolve_claude_model(current) == current


def test_build_api_body_remaps_stale_model(proxy_module, monkeypatch):
    monkeypatch.setattr(proxy.auth, "body_template", {"max_tokens": 1})
    monkeypatch.setattr(proxy, "_cached_billing_block", None)
    monkeypatch.setattr(proxy, "_cached_auth_blocks", [])
    monkeypatch.setattr(proxy, "reasoning_override_enabled", False)
    body = _build_api_body(
        None, [{"role": "user", "content": "hi"}], "claude-sonnet-4-6", 4096
    )
    assert body["model"] == "claude-sonnet-5-5"
