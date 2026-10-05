"""Direct (non-CLI) Codex transport tests. No real network or credentials."""

import asyncio
import json
import logging
import os
import stat

import pytest

from proxy_internal.auth.codex import CodexAuthCache
from proxy_internal.providers import codex as codex_provider


def _auth_file(home, tokens, last_refresh="2030-01-01T00:00:00+00:00", extra=None):
    data = {"tokens": dict(tokens), "last_refresh": last_refresh}
    if extra:
        data.update(extra)
    path = os.path.join(home, ".codex", "auth.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.chmod(path, 0o600)
    return path


def _sse(*payloads):
    return "".join(f"data: {p}\n\n" for p in payloads)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_endpoint_and_headers(monkeypatch, tmp_path):
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    _auth_file(home, {"access_token": "AT", "refresh_token": "RT", "account_id": "ACC"})
    cache = CodexAuthCache()
    assert cache.reload_from_disk_if_stale() is True
    headers = cache.get_chatgpt_headers()
    for key in (
        "Authorization",
        "chatgpt-account-id",
        "originator",
        "User-Agent",
        "OpenAI-Beta",
        "Accept",
        "Content-Type",
        "session_id",
        "x-client-request-id",
    ):
        assert key in headers, f"missing header {key}"
    assert headers["Authorization"] == "Bearer AT"
    assert headers["chatgpt-account-id"] == "ACC"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")
    assert headers["Accept"] == "text/event-stream"
    assert headers["OpenAI-Beta"] == "responses=v1"

    seen = {}

    async def fake_post(headers, body):
        seen["headers"] = headers
        seen["body"] = body
        assert body["store"] is False  # server-enforced; never omitted
        assert "messages" not in body
        assert isinstance(body["instructions"], str)
        assert isinstance(body["input"], list)
        assert body["input"][0]["content"][0]["type"] == "input_text"
        payload = _sse(
            json.dumps({"type": "response.output_text.delta", "delta": "hello"}),
            json.dumps({"type": "response.completed",
                        "response": {"usage": {"input_tokens": 3, "output_tokens": 5}}}),
        )
        return 200, payload

    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)

    import proxy

    monkeypatch.setattr(proxy, "codex_auth", cache)
    assert codex_provider.CHATGPT_CODEX_RESPONSES_URL == \
        "https://chatgpt.com/backend-api/codex/responses"
    text, usage = _run(codex_provider._call_codex_direct_http(
        "sys", [{"role": "user", "content": "hi"}], "gpt-6-sol", 64))
    assert text == "hello"
    assert usage == {"input_tokens": 3, "output_tokens": 5}


def test_delta_mapping_reasoning_and_nonstreaming_agg(monkeypatch, tmp_path):
    async def fake_post(headers, body):
        payload = _sse(
            json.dumps({"type": "response.output_text.delta", "delta": "Hel"}),
            json.dumps({"type": "response.reasoning_text.delta", "delta": "secret-thought"}),
            json.dumps({"type": "response.reasoning_summary_text.delta", "delta": "sum"}),
            json.dumps({"type": "response.output_text.delta", "delta": "lo"}),
            json.dumps({"type": "response.completed",
                        "response": {"usage": {"input_tokens": 1, "output_tokens": 2}}}),
        )
        return 200, payload

    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)
    cache = CodexAuthCache()
    cache.access_token = "AT"
    cache.account_id = "ACC"
    cache.expires_at = __import__("datetime").datetime.now() + \
        __import__("datetime").timedelta(hours=1)
    import proxy
    monkeypatch.setattr(proxy, "codex_auth", cache)

    text, _ = _run(codex_provider._call_codex_direct_http(
        None, [{"role": "user", "content": "hi"}], "m", 16))
    assert text == "Hello"  # reasoning deltas scrubbed, not merged
    assert "secret-thought" not in text

    async def collect():
        out = []
        async for chunk in codex_provider.call_codex_streaming(None, [{"role": "user", "content": "hi"}], "m", 16):
            out.append(chunk)
        return "".join(out)

    streamed = _run(collect())
    assert "Hello" in streamed


def test_store_false_400_falls_back_once(monkeypatch, caplog):
    calls = {"direct": 0, "cli": 0}

    async def fake_post(headers, body):
        calls["direct"] += 1
        return 400, "Store must be set to false"

    async def fake_cli(system_prompt, messages, model, max_tokens):
        calls["cli"] += 1
        return "cli-text"

    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)
    monkeypatch.setattr(codex_provider, "_call_codex_direct_cli", fake_cli)
    cache = CodexAuthCache()
    import datetime as _dt
    cache.access_token = "AT"
    cache.expires_at = _dt.datetime.now() + _dt.timedelta(hours=1)
    import proxy
    monkeypatch.setattr(proxy, "codex_auth", cache)

    with caplog.at_level(logging.WARNING, logger="proxy"):
        text = _run(codex_provider.call_codex_direct(None, [{"role": "user", "content": "x"}], "m", 8))
    assert text == "cli-text"
    assert calls == {"direct": 1, "cli": 1}  # no direct retry
    assert any("falling back" in r.message.lower() for r in caplog.records)


def test_401_falls_back_to_cli_once(monkeypatch):
    calls = {"direct": 0, "cli": 0}

    async def fake_post(headers, body):
        calls["direct"] += 1
        return 401, "unauthorized"

    async def fake_cli(system_prompt, messages, model, max_tokens):
        calls["cli"] += 1
        return "cli-text"

    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)
    monkeypatch.setattr(codex_provider, "_call_codex_direct_cli", fake_cli)
    cache = CodexAuthCache()
    import datetime as _dt
    cache.access_token = "AT"
    cache.expires_at = _dt.datetime.now() + _dt.timedelta(hours=1)
    import proxy
    monkeypatch.setattr(proxy, "codex_auth", cache)

    text = _run(codex_provider.call_codex_direct(None, [{"role": "user", "content": "x"}], "m", 8))
    assert text == "cli-text"
    assert calls == {"direct": 1, "cli": 1}


def test_malformed_sse_falls_back(monkeypatch):
    calls = {"cli": 0}

    async def fake_post(headers, body):
        return 200, "data: {not-json\n\ndata: garbage{{{\n\n"

    async def fake_cli(system_prompt, messages, model, max_tokens):
        calls["cli"] += 1
        return "cli-text"

    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)
    monkeypatch.setattr(codex_provider, "_call_codex_direct_cli", fake_cli)
    cache = CodexAuthCache()
    import datetime as _dt
    cache.access_token = "AT"
    cache.expires_at = _dt.datetime.now() + _dt.timedelta(hours=1)
    import proxy
    monkeypatch.setattr(proxy, "codex_auth", cache)

    text = _run(codex_provider.call_codex_direct(None, [{"role": "user", "content": "x"}], "m", 8))
    assert text == "cli-text"
    assert calls["cli"] == 1


def test_refresh_rotation_persists_and_preserves(monkeypatch, tmp_path):
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    path = _auth_file(
        home,
        {"access_token": "OLD", "refresh_token": "OLD-RT", "account_id": "ACC"},
        last_refresh="2020-01-01T00:00:00+00:00",
        extra={"custom_field": {"keep": True}},
    )
    cache = CodexAuthCache()
    cache.access_token = "OLD"
    cache.refresh_token = "OLD-RT"
    cache.account_id = "ACC"
    import datetime as _dt
    cache.expires_at = _dt.datetime.now() - _dt.timedelta(seconds=1)

    async def fake_refresh(self):
        self.access_token = "NEW-AT"
        self.refresh_token = "NEW-RT"
        self.expires_at = _dt.datetime.now() + _dt.timedelta(hours=1)
        return True

    monkeypatch.setattr(CodexAuthCache, "_do_refresh", fake_refresh)
    assert _run(cache.refresh_if_needed()) is True
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["tokens"]["access_token"] == "NEW-AT"
    assert data["tokens"]["refresh_token"] == "NEW-RT"
    assert data["tokens"]["account_id"] == "ACC"
    assert data["custom_field"] == {"keep": True}  # unknown fields preserved
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_stale_cache_adopts_disk_without_network(monkeypatch, tmp_path):
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    _auth_file(home, {"access_token": "DISK-AT", "refresh_token": "DISK-RT",
                       "account_id": "DISK-ACC"})
    cache = CodexAuthCache()
    cache.access_token = "STALE"
    import datetime as _dt
    cache.expires_at = _dt.datetime.now() - _dt.timedelta(seconds=1)
    cache.refresh_token = "STALE-RT"

    async def no_network(*a, **k):
        raise AssertionError("network must not be touched")

    monkeypatch.setattr(CodexAuthCache, "_do_refresh", no_network)
    assert _run(cache.refresh_if_needed()) is True
    assert cache.access_token == "DISK-AT"
    assert cache.account_id == "DISK-ACC"


def test_no_credential_in_logs(monkeypatch, tmp_path, caplog):
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    secret_at = "SECRET-ACCESS-xyz"
    secret_rt = "SECRET-REFRESH-xyz"
    _auth_file(home, {"access_token": "DISK-AT", "refresh_token": secret_rt,
                       "account_id": "ACC"})
    cache = CodexAuthCache()
    cache.access_token = secret_at
    cache.refresh_token = secret_rt
    import datetime as _dt
    cache.expires_at = _dt.datetime.now() - _dt.timedelta(seconds=1)

    async def fake_do_refresh(self):
        self.access_token = "NEW"
        self.refresh_token = "NEW-RT"
        self.expires_at = _dt.datetime.now() + _dt.timedelta(hours=1)
        return True

    async def fake_post(headers, body):
        return 500, "boom"

    async def fake_cli(system_prompt, messages, model, max_tokens):
        return "cli-text"

    monkeypatch.setattr(CodexAuthCache, "_do_refresh", fake_do_refresh)
    monkeypatch.setattr(codex_provider, "_direct_post", fake_post)
    monkeypatch.setattr(codex_provider, "_call_codex_direct_cli", fake_cli)
    import proxy
    monkeypatch.setattr(proxy, "codex_auth", cache)

    with caplog.at_level(logging.DEBUG, logger="proxy"):
        _run(cache.refresh_if_needed())
        _run(codex_provider.call_codex_direct(None, [{"role": "user", "content": "x"}], "m", 8))

    blob = "\n".join(f"{r.levelname} {r.message}" for r in caplog.records)
    assert secret_at not in blob
    assert secret_rt not in blob
