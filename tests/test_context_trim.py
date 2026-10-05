"""Context-window rejection handling: detection, trimming, and Z.AI retry.

Every upstream is faked — no network, no live credentials.
"""
import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy
import proxy_internal.providers.zai as zai_mod
from proxy_internal.context_trim import (
    MAX_TRIM_ROUNDS,
    ContextExceededError,
    context_excess_tokens,
    trim_messages_for_context,
)


def _context_error_body(input_tokens=200_000, output_tokens=0, max_context_tokens=190_000):
    return json.dumps({
        "error": {
            "code": "context_length_exceeded",
            "message": "Prompt is too long",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_context_tokens": max_context_tokens,
        }
    })


# --- fakes -----------------------------------------------------------------

class _FakeResp:
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._json = json_data
        self._text = text_data

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _FakeCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class _ScriptedSession:
    """Serves one pre-scripted response per POST and records every payload."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeCM(self._responses.pop(0))

    async def close(self):
        pass


@pytest.fixture()
def zai_env(monkeypatch):
    """Z.AI key set + a shared-session slot the tests fill with a fake."""
    monkeypatch.setattr(proxy, "zai_api_key", "k-test", raising=False)
    if getattr(getattr(proxy, "auth", None), "session", None) is not None:
        monkeypatch.setattr(proxy.auth, "session", None, raising=False)
    monkeypatch.setattr(proxy, "third_party_session", None, raising=False)
    return monkeypatch


def _use_session(monkeypatch, session):
    monkeypatch.setattr(proxy, "third_party_session", session, raising=False)


def _ok_resp(text="ok"):
    return _FakeResp(status=200, json_data={"choices": [{"message": {"content": text}}]})


def _messages_with_one_big_message():
    return [
        {"role": "system", "content": "s" * 40},
        {"role": "user", "content": "old context " * 500},
        {"role": "user", "content": "the turn to answer"},
    ]


def _contents(payload):
    return [m["content"] for m in payload["messages"]]


# --- detection -------------------------------------------------------------

class TestContextExcessTokens:
    def test_reads_the_overage_from_a_context_rejection(self):
        assert context_excess_tokens(400, _context_error_body(200_000, 0, 190_000)) == 10_000
        assert context_excess_tokens(400, _context_error_body(200_000, 500, 190_000)) == 10_500

    def test_non_400_status_is_not_a_context_rejection(self):
        assert context_excess_tokens(500, _context_error_body()) is None
        assert context_excess_tokens(200, "context_length_exceeded") is None

    def test_other_error_codes_are_ignored(self):
        body = json.dumps({"error": {"code": "rate_limited", "message": "context_length_exceeded"}})
        assert context_excess_tokens(400, body) is None

    def test_malformed_or_unusable_bodies_are_ignored(self):
        assert context_excess_tokens(400, "context_length_exceeded") is None
        assert context_excess_tokens(400, "{not json context_length_exceeded") is None
        assert context_excess_tokens(400, json.dumps({"error": {"code": "context_length_exceeded"}})) is None
        assert context_excess_tokens(
            400, _context_error_body(180_000, 0, 200_000)
        ) is None  # under the limit: nothing to trim


# --- trimming --------------------------------------------------------------

class TestTrimMessages:
    def test_drops_the_oldest_droppable_message(self):
        messages = _messages_with_one_big_message()
        trimmed, removed = trim_messages_for_context(messages, 1_000)
        assert _contents({"messages": trimmed}) == ["s" * 40, "the turn to answer"]
        assert removed == len("old context " * 500)

    def test_drops_only_until_the_overage_is_freed(self):
        messages = [
            {"role": "system", "content": "diary"},
            {"role": "user", "content": "a" * 5_000},
            {"role": "user", "content": "b" * 5_000},
            {"role": "user", "content": "latest"},
        ]
        trimmed, removed = trim_messages_for_context(messages, 100)
        assert [m["content"] for m in trimmed] == ["diary", "b" * 5_000, "latest"]
        assert removed == 5_000

    def test_protects_the_first_system_and_final_messages(self):
        messages = [
            {"role": "system", "content": "diary"},
            {"role": "user", "content": "a" * 5_000},
            {"role": "user", "content": "b" * 5_000},
            {"role": "assistant", "content": "c" * 5_000},
            {"role": "user", "content": "latest"},
        ]
        trimmed, removed = trim_messages_for_context(messages, 100_000)
        assert [m["content"] for m in trimmed] == ["diary", "latest"]
        assert removed == 15_000

    def test_skips_system_messages_when_choosing_victims(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "mid-prompt"},
            {"role": "user", "content": "d" * 5_000},
            {"role": "user", "content": "latest"},
        ]
        trimmed, removed = trim_messages_for_context(messages, 100)
        assert [m["content"] for m in trimmed] == ["first", "mid-prompt", "latest"]
        assert removed == 5_000

    def test_reports_zero_when_nothing_may_be_dropped(self):
        messages = [{"role": "system", "content": "diary"}, {"role": "user", "content": "latest"}]
        trimmed, removed = trim_messages_for_context(messages, 100)
        assert trimmed == messages
        assert removed == 0

    def test_counts_multimodal_content(self):
        messages = [
            {"role": "system", "content": "diary"},
            {"role": "user", "content": [{"type": "text", "text": "e" * 900}]},
            {"role": "user", "content": "latest"},
        ]
        _trimmed, removed = trim_messages_for_context(messages, 10)
        assert removed == 900


# --- Z.AI wiring -----------------------------------------------------------

class TestZaiDirectTrimming:
    @pytest.mark.asyncio
    async def test_retries_with_older_history_dropped(self, zai_env):
        sess = _ScriptedSession([_FakeResp(status=400, text_data=_context_error_body()), _ok_resp("answer")])
        _use_session(zai_env, sess)

        out = await zai_mod.call_zai_direct(None, _messages_with_one_big_message(), "zai:glm-4.5", 128)

        assert out == "answer"
        assert len(sess.payloads) == 2
        assert _contents(sess.payloads[0]) == ["s" * 40, "old context " * 500, "the turn to answer"]
        assert _contents(sess.payloads[1]) == ["s" * 40, "the turn to answer"]

    @pytest.mark.asyncio
    async def test_no_droppable_history_fails_without_a_retry(self, zai_env):
        sess = _ScriptedSession([_FakeResp(status=400, text_data=_context_error_body())])
        _use_session(zai_env, sess)

        with pytest.raises(HTTPException) as excinfo:
            await zai_mod.call_zai_direct(
                None,
                [{"role": "system", "content": "diary"}, {"role": "user", "content": "latest"}],
                "zai:glm-4.5",
                128,
            )

        assert excinfo.value.status_code == 400
        assert "no older history" in excinfo.value.detail
        assert len(sess.payloads) == 1

    @pytest.mark.asyncio
    async def test_trimming_rounds_are_bounded(self, zai_env):
        # Six droppable messages and a small overage: each round frees some
        # history, so the loop runs to its cap rather than forever.
        messages = [{"role": "system", "content": "diary"}]
        messages += [{"role": "user", "content": f"h{i}" + "x" * 3_000} for i in range(6)]
        messages.append({"role": "user", "content": "latest"})
        sess = _ScriptedSession([_FakeResp(status=400, text_data=_context_error_body(191_000, 0, 190_000))] * 8)
        _use_session(zai_env, sess)

        with pytest.raises(HTTPException) as excinfo:
            await zai_mod.call_zai_direct(None, messages, "zai:glm-4.5", 128)

        assert excinfo.value.status_code == 400
        assert "even after dropping older history" in excinfo.value.detail
        assert len(sess.payloads) == MAX_TRIM_ROUNDS + 1


class TestZaiStreamingTrimming:
    @pytest.mark.asyncio
    async def test_stream_retries_with_older_history_dropped(self, zai_env):
        sess = _ScriptedSession([_FakeResp(status=400, text_data=_context_error_body()), _ok_resp("streamed")])
        _use_session(zai_env, sess)

        events = [
            event
            async for event in zai_mod.call_zai_streaming(
                None, _messages_with_one_big_message(), "zai:glm-4.5", 128
            )
        ]

        assert any("streamed" in event for event in events)
        assert events[-1].strip() == "data: [DONE]"
        assert len(sess.payloads) == 2
        assert _contents(sess.payloads[1]) == ["s" * 40, "the turn to answer"]

    @pytest.mark.asyncio
    async def test_stream_reports_when_nothing_can_be_dropped(self, zai_env):
        sess = _ScriptedSession([_FakeResp(status=400, text_data=_context_error_body())])
        _use_session(zai_env, sess)

        events = [
            event
            async for event in zai_mod.call_zai_streaming(
                None,
                [{"role": "system", "content": "diary"}, {"role": "user", "content": "latest"}],
                "zai:glm-4.5",
                128,
            )
        ]

        assert any("Error 400" in event for event in events)
        assert events[-1].strip() == "data: [DONE]"
        assert len(sess.payloads) == 1


class TestContextExceededErrorShape:
    def test_excess_tokens_are_clamped_and_readable(self):
        error = ContextExceededError(12)
        assert error.excess_tokens == 12
        assert "12 tokens" in str(error)
        assert ContextExceededError(-5).excess_tokens == 0
