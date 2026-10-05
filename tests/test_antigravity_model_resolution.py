"""Antigravity model resolution: prefixed external names resolve to bare upstream IDs."""
import pytest

from proxy_internal.model_detect import is_antigravity_model
from proxy_internal.providers.antigravity import _get_antigravity_model_id


@pytest.mark.parametrize("model,expected", [
    ("antigravity-claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("antigravity-claude-opus-4-6", "claude-opus-4-6-thinking"),
    ("antigravity-claude-opus-4-6-thinking", "claude-opus-4-6-thinking"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-opus-4-6-thinking", "claude-opus-4-6-thinking"),
    ("antigravity-gpt-oss-120b-medium", "gpt-oss-120b-medium"),
    ("gpt-oss-120b-medium", "gpt-oss-120b-medium"),
    ("antigravity-gemini-3.8-flash", "gemini-3.8-flash-high"),
    ("gemini-3.8-flash", "gemini-3.8-flash-high"),
    ("antigravity-gemini-3.8-flash-high", "gemini-3.8-flash-high"),
    ("gemini-3.8-flash-low", "gemini-3.8-flash-low"),
    ("gemini-3.8-flash-medium", "gemini-3.8-flash-medium"),
    ("gemini-3.8-flash-tiered", "gemini-3.8-flash-tiered"),
    ("antigravity-gemini-3.1-pro", "gemini-pro-agent"),
    ("gemini-3.1-pro", "gemini-pro-agent"),
    ("gemini-3.1-pro-preview", "gemini-pro-agent"),
    ("gemini-3.1-pro-high", "gemini-pro-agent"),
    ("antigravity-gemini-3.1-pro-high", "gemini-pro-agent"),
    ("gemini-3.1-pro-low", "gemini-3.1-pro-low"),
    ("gemini-pro-agent", "gemini-pro-agent"),
    ("antigravity-gemini-3-flash", "gemini-3-flash"),
    ("antigravity-some-unknown-xyz", "some-unknown-xyz"),
])
def test_antigravity_model_id_resolution(model, expected):
    assert _get_antigravity_model_id(model) == expected


def test_opus_never_resolves_to_45():
    for name in ("antigravity-claude-opus-4-6", "antigravity-claude-opus-4-6-thinking",
                 "claude-opus-4-6", "claude-opus-4-6-thinking"):
        assert "4-5" not in _get_antigravity_model_id(name)


@pytest.mark.parametrize("model", [
    "antigravity-claude-sonnet-4-6",
    "antigravity-claude-opus-4-6-thinking",
    "antigravity-gemini-3.8-flash",
    "antigravity-gpt-oss-120b-medium",
])
def test_prefixed_names_route_to_antigravity(model):
    assert is_antigravity_model(model)


# Note: bare gemini-3/gemini-2.5/gpt-oss- names route to Antigravity via the
# pre-existing is_antigravity_model detector (unchanged here); only bare
# Claude-style names are isolated from Antigravity by the routing prefix.
@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
])
def test_bare_claude_does_not_route_to_antigravity(model):
    assert not is_antigravity_model(model)


def test_structured_resolver_uses_same_mapping_no_network(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import proxy

    fake_account = SimpleNamespace(
        session=None,
        project_id="fixture-project",
        get_auth_headers=lambda: {"Authorization": "Bearer fixture"},
    )
    fake_auth = SimpleNamespace(
        is_ready=True,
        native_keychain=False,
        session=None,
        refresh_if_needed=lambda: False,
        get_next_account=lambda: fake_account,
    )
    monkeypatch.setattr(proxy, "antigravity_auth", fake_auth)
    monkeypatch.setattr(proxy, "create_session", lambda: SimpleNamespace())

    async def go(model):
        return await proxy._resolve_oai_compatible(model)
    loop = asyncio.new_event_loop()
    try:
        resolved = loop.run_until_complete(go("antigravity-claude-opus-4-6"))
        resolved_gem = loop.run_until_complete(go("antigravity-gemini-3.8-flash"))
        bare = loop.run_until_complete(go("claude-sonnet-4-6"))
    finally:
        loop.close()
    assert resolved["api_model"] == "claude-opus-4-6-thinking"
    assert resolved["provider_name"] == "Antigravity"
    assert resolved["format"] == "gemini"
    assert resolved_gem["api_model"] == "gemini-3.8-flash-high"
    assert bare is None
