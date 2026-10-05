"""Regression tests for the Claude per-model output-token clamp.

The auth-capture template is captured once — with ``proxy.DEFAULT_MODEL`` whose
ceiling is 128K — and then replayed for every model routed to the Claude
provider. A request routed to claude-haiku-4-5 (64K ceiling) therefore carried
``max_tokens: 128000`` and Anthropic rejected it outright:

    max_tokens: 128000 > 64000, which is the maximum allowed number of output
    tokens for claude-haiku-4-5-20251001

No live network here: the passthrough test drives a fake aiohttp session and
asserts on the JSON body that would hit the wire.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy  # noqa: E402
import proxy_internal.providers.claude as claude_mod  # noqa: E402
from proxy_internal.providers.claude import (  # noqa: E402
    _build_api_body,
    _clamp_max_tokens,
)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5-5"


class _FakeResp:
    def __init__(self, payload):
        self.status = 200
        self._raw = json.dumps(payload).encode()

    async def read(self):
        return self._raw


class _FakeCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Captures the bytes the provider would POST to api.anthropic.com."""

    def __init__(self, payload):
        self._payload = payload
        self.sent = None

    def post(self, url, data=None, headers=None):
        self.sent = data
        return _FakeCM(_FakeResp(self._payload))

    async def close(self):
        pass


class TestClampMaxTokens:
    def test_dated_haiku_snapshot_clamped(self):
        body = {"max_tokens": 128000}
        _clamp_max_tokens(body, HAIKU)
        assert body["max_tokens"] == 64000

    def test_undated_haiku_alias_clamped(self):
        body = {"max_tokens": 128000}
        _clamp_max_tokens(body, "claude-haiku-4-5")
        assert body["max_tokens"] == 64000

    def test_below_ceiling_left_alone(self):
        body = {"max_tokens": 32000}
        _clamp_max_tokens(body, HAIKU)
        assert body["max_tokens"] == 32000

    def test_unlisted_model_left_alone(self):
        body = {"max_tokens": 128000}
        _clamp_max_tokens(body, SONNET)
        assert body["max_tokens"] == 128000

    def test_absent_or_non_int_is_noop(self):
        for value in ({}, {"max_tokens": None}, {"max_tokens": "128000"}):
            body = dict(value)
            _clamp_max_tokens(body, HAIKU)
            assert body == value


class TestBuildApiBodyClamp:
    """The template's 128K value must not survive into a haiku request."""

    @pytest.fixture()
    def template_128k(self, proxy_module, monkeypatch):
        monkeypatch.setattr(
            proxy.auth,
            "body_template",
            {"model": SONNET, "max_tokens": 128000, "stream": False},
        )
        monkeypatch.setattr(proxy, "_cached_billing_block", None)
        monkeypatch.setattr(proxy, "_cached_auth_blocks", [])
        monkeypatch.setattr(proxy, "reasoning_override_enabled", False)

    def test_haiku_body_clamped(self, template_128k):
        body = _build_api_body(
            None, [{"role": "user", "content": "hi"}], HAIKU, 4096
        )
        assert body["max_tokens"] == 64000

    def test_sonnet_body_unchanged(self, template_128k):
        body = _build_api_body(
            None, [{"role": "user", "content": "hi"}], SONNET, 4096
        )
        assert body["max_tokens"] == 128000


class TestPassthroughClamp:
    """Route B forwards the caller's max_tokens verbatim — clamp it too."""

    @pytest.mark.asyncio
    async def test_passthrough_sends_clamped_max_tokens(
        self, proxy_module, monkeypatch
    ):
        session = _FakeSession({"content": [{"type": "text", "text": "ok"}]})
        monkeypatch.setattr(proxy.auth, "headers", {"x-api-key": "stub"})
        monkeypatch.setattr(proxy.auth, "body_template", {"max_tokens": 1})
        monkeypatch.setattr(proxy.auth, "session", session)
        monkeypatch.setattr(proxy, "_cached_billing_block", None)
        monkeypatch.setattr(proxy, "_cached_auth_blocks", [])
        monkeypatch.setattr(proxy, "reasoning_override_enabled", False)

        result = await claude_mod.call_claude_messages_passthrough(
            {
                "model": HAIKU,
                "max_tokens": 128000,
                "messages": [{"role": "user", "content": "hi"}],
            },
            False,
        )

        assert json.loads(session.sent)["max_tokens"] == 64000
        assert result["content"][0]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_passthrough_keeps_sonnet_max_tokens(
        self, proxy_module, monkeypatch
    ):
        session = _FakeSession({"content": []})
        monkeypatch.setattr(proxy.auth, "headers", {"x-api-key": "stub"})
        monkeypatch.setattr(proxy.auth, "body_template", {"max_tokens": 1})
        monkeypatch.setattr(proxy.auth, "session", session)
        monkeypatch.setattr(proxy, "_cached_billing_block", None)
        monkeypatch.setattr(proxy, "_cached_auth_blocks", [])
        monkeypatch.setattr(proxy, "reasoning_override_enabled", False)

        await claude_mod.call_claude_messages_passthrough(
            {
                "model": SONNET,
                "max_tokens": 128000,
                "messages": [{"role": "user", "content": "hi"}],
            },
            False,
        )

        assert json.loads(session.sent)["max_tokens"] == 128000
