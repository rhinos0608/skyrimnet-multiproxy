"""Isolated tests for the official Command Code Provider API route.

Uses mocked aiohttp sessions and a fake CLI — no live network, no real
credentials. Covers `command_api:` direct/streaming, exact-403 fallback
semantics, timeouts, key handling, and absence of write/shell CLI flags.
"""
import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy
import proxy_internal.providers.command_code as cmd_mod


class _AsyncLines:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        async def _gen():
            for line in self._lines:
                yield line
        return _gen()


class _FakeResp:
    def __init__(self, status=200, json_data=None, text_data="",
                 content_lines=None):
        self.status = status
        self._json = json_data
        self._text = text_data
        self.content = _AsyncLines(content_lines or [])

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _FakeCM:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self._resp

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        if self._exc is not None:
            raise self._exc
        return _FakeCM(self._resp)

    async def close(self):
        self.closed = True


def _ok_resp(text="hello from api"):
    return _FakeResp(
        status=200,
        json_data={"choices": [{"message": {"content": text}}]},
    )


@pytest.fixture()
def api_env(monkeypatch):
    """Route _command_api_session at the shared-session branch (no ownership)."""
    sess_holder = {}
    monkeypatch.setattr(proxy, "third_party_session", None, raising=False)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", None, raising=False)
    if hasattr(proxy, "auth") and getattr(proxy.auth, "session", None) is not None:
        monkeypatch.setattr(proxy.auth, "session", None, raising=False)
    return sess_holder


def _use_session(monkeypatch, session):
    monkeypatch.setattr(proxy, "third_party_session", session, raising=False)


# --- direct success -------------------------------------------------------

@pytest.mark.asyncio
async def test_direct_success(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp("api-text"))
    _use_session(monkeypatch, sess)
    out = await cmd_mod.call_command_code_api_direct(
        "sys", [{"role": "user", "content": "hi"}],
        "command_api:zai/glm-5", 64, api_key="k-test",
    )
    assert out == "api-text"
    call = sess.calls[0]
    assert call["url"] == cmd_mod.COMMAND_CODE_API_CHAT_URL
    assert call["json"]["model"] == "zai/glm-5"
    assert "stream" not in call["json"]


@pytest.mark.asyncio
async def test_direct_headers_minimal_no_fakes(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp())
    _use_session(monkeypatch, sess)
    await cmd_mod.call_command_code_api_direct(
        None, [{"role": "user", "content": "hi"}],
        "command_api:zai/glm-5", 32, api_key="k-secret",
    )
    headers = sess.calls[0]["headers"]
    assert set(headers) == {"Authorization", "Content-Type"}
    assert headers["Authorization"] == "Bearer k-secret"


# --- model-id validation ---------------------------------------------------

@pytest.mark.asyncio
async def test_short_id_rejected_without_network(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp())
    _use_session(monkeypatch, sess)
    with pytest.raises(HTTPException) as ei:
        await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}],
            "command_api:claude-opus", 32, api_key="k",
        )
    assert ei.value.status_code == 400
    assert sess.calls == []


# --- key handling ----------------------------------------------------------

@pytest.mark.asyncio
async def test_key_from_proxy_attr(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp("via-attr"))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "command_code_api_key", "attr-key", raising=False)
    monkeypatch.setattr(proxy, "_resolve_command_code_api_key", lambda: None, raising=False)
    out = await cmd_mod.call_command_code_api_direct(
        None, [{"role": "user", "content": "hi"}], "command_api:zai/glm-5", 16,
    )
    assert out == "via-attr"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer attr-key"


@pytest.mark.asyncio
async def test_dynamic_resolver_beats_stale_attr(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp("via-rotated"))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "command_code_api_key", "stale-attr", raising=False)
    monkeypatch.setattr(proxy, "_resolve_command_code_api_key", lambda: "fresh-pi-key", raising=False)
    out = await cmd_mod.call_command_code_api_direct(
        None, [{"role": "user", "content": "hi"}], "command_api:zai/glm-5", 16,
    )
    assert out == "via-rotated"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer fresh-pi-key"


@pytest.mark.asyncio
async def test_missing_key_503(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp())
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "command_code_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "_resolve_command_code_api_key", lambda: None, raising=False)
    with pytest.raises(HTTPException) as ei:
        await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}], "command_api:zai/glm-5", 16,
        )
    assert ei.value.status_code == 503
    assert sess.calls == []


# --- 403 upgrade_required semantics ----------------------------------------

def _upgrade_body():
    return json.dumps({"code": "upgrade_required", "message": "Go $1 cannot use this"})


@pytest.mark.asyncio
async def test_403_upgrade_exact_falls_back_to_cli(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        out = await cmd_mod.call_command_code_api_direct(
            "sys", [{"role": "user", "content": "hi"}],
            "command_api:anyvendor/glm-5", 64, api_key="k",
        )
    assert out == "cli-text"
    assert cli.await_args[0][2] == "command_code:anyvendor/glm-5"


@pytest.mark.asyncio
async def test_403_upgrade_nested_error_shape_falls_back(api_env, monkeypatch):
    body = json.dumps({"error": {"code": "upgrade_required"}})
    sess = _FakeSession(_FakeResp(status=403, text_data=body))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        out = await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}],
            "command_api:anyvendor/glm-5", 64, api_key="k",
        )
    assert out == "cli-text"
    assert cli.await_count == 1


@pytest.mark.asyncio
async def test_403_upgrade_non_allowlisted_no_fallback(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        with pytest.raises(HTTPException) as ei:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:unknownvendor/unknown-model-xyz", 64, api_key="k",
            )
    assert ei.value.status_code == 403
    assert "GOAT" in ei.value.detail
    assert cli.await_count == 0


@pytest.mark.asyncio
async def test_403_upgrade_cli_missing_actionable(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", None, raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        with pytest.raises(HTTPException) as ei:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:anyvendor/glm-5", 64, api_key="k",
            )
    assert ei.value.status_code == 403
    assert "CLI" in ei.value.detail
    assert cli.await_count == 0


@pytest.mark.asyncio
async def test_403_other_code_no_fallback(api_env, monkeypatch):
    body = json.dumps({"code": "forbidden", "message": "denied"})
    sess = _FakeSession(_FakeResp(status=403, text_data=body))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        with pytest.raises(HTTPException) as ei:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:anyvendor/glm-5", 64, api_key="k",
            )
    assert ei.value.status_code == 403
    assert ei.value.detail == "Command Code API request forbidden"
    assert "denied" not in ei.value.detail
    assert cli.await_count == 0


@pytest.mark.asyncio
async def test_cli_direct_unavailable_503(api_env):
    with pytest.raises(HTTPException) as ei:
        await cmd_mod.call_command_code_direct(
            None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
        )
    assert ei.value.status_code == 503


# --- error redaction --------------------------------------------------------

@pytest.mark.asyncio
async def test_upstream_error_body_not_leaked(api_env, monkeypatch):
    big = "X" * 2000 + "secret-sauce"
    sess = _FakeSession(_FakeResp(status=500, text_data=big))
    _use_session(monkeypatch, sess)
    with pytest.raises(HTTPException) as ei:
        await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}],
            "command_api:zai/glm-5", 16, api_key="k-cred",
        )
    assert ei.value.status_code == 500
    assert "secret-sauce" not in ei.value.detail
    assert "k-cred" not in ei.value.detail


# --- streaming ---------------------------------------------------------------

def _sse_resp(chunks):
    lines = []
    for c in chunks:
        lines.append(('data: {"choices":[{"delta":{"content":"%s"}}]}' % c).encode())
    lines.append(b"data: [DONE]")
    return _FakeResp(status=200, content_lines=lines)


@pytest.mark.asyncio
async def test_streaming_sse_passthrough(api_env, monkeypatch):
    sess = _FakeSession(_sse_resp(["he", "llo"]))
    _use_session(monkeypatch, sess)
    events = []
    async for chunk in cmd_mod.call_command_code_api_streaming(
        None, [{"role": "user", "content": "hi"}],
        "command_api:zai/glm-5", 64, api_key="k",
    ):
        events.append(chunk)
    assert any("he" in e for e in events)
    assert events[-1] == "data: [DONE]\n\n"
    assert sess.calls[0]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_streaming_403_upgrade_falls_back_to_cli(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)

    seen = {}

    async def _fake_cli_stream(*args, **kwargs):
        seen['model'] = args[2] if len(args) > 2 else kwargs.get('model')
        yield 'data: {"choices":[{"delta":{"content":"cli"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    with patch.object(cmd_mod, 'call_command_code_streaming',
                      new=_fake_cli_stream):
        events = [c async for c in cmd_mod.call_command_code_api_streaming(
            None, [{'role': 'user', 'content': 'hi'}],
            'command_api:anyvendor/glm-5', 64, api_key='k',
        )]
    assert any('cli' in e for e in events)
    assert seen.get('model') == 'command_code:anyvendor/glm-5'


@pytest.mark.asyncio
async def test_streaming_403_other_no_fallback(api_env, monkeypatch):
    body = json.dumps({"code": "access-denied-xyz"})
    sess = _FakeSession(_FakeResp(status=403, text_data=body))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_streaming") as cli:
        events = [c async for c in cmd_mod.call_command_code_api_streaming(
            None, [{"role": "user", "content": "hi"}],
            "command_api:anyvendor/glm-5", 64, api_key="k",
        )]
        cli.assert_not_called()
    assert events[-1] == "data: [DONE]\n\n"
    assert "access-denied-xyz" not in "".join(events)


# --- timeouts ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_direct_timeout_504(api_env, monkeypatch):
    sess = _FakeSession(_ok_resp())
    _use_session(monkeypatch, sess)
    with patch.object(proxy, "_with_retry", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        with pytest.raises(HTTPException) as ei:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:zai/glm-5", 16, api_key="k",
            )
    assert ei.value.status_code == 504


@pytest.mark.asyncio
async def test_streaming_connect_timeout_yields_error(api_env, monkeypatch):
    sess = _FakeSession(exc=asyncio.TimeoutError())
    _use_session(monkeypatch, sess)
    events = [c async for c in cmd_mod.call_command_code_api_streaming(
        None, [{"role": "user", "content": "hi"}],
        "command_api:zai/glm-5", 16, api_key="k",
    )]
    assert any("timeout" in e.lower() for e in events)
    assert events[-1] == "data: [DONE]\n\n"


# --- CLI flag hygiene ------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_paths_have_no_write_shell_flags(api_env, monkeypatch):
    captured = []
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        captured.append(list(cmd_args))
        return (b'[{"type":"text","text":"ok"}]', b"")

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        await cmd_mod.call_command_code_direct(
            None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
        )
        async for _ in cmd_mod.call_command_code_streaming(
            None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
        ):
            pass
    assert len(captured) == 2
    for args in captured:
        assert "--print" in args
        assert "--yolo" not in args
        assert not any("dangerously" in a for a in args)


@pytest.mark.asyncio
async def test_upstream_body_prefix_not_in_logs(api_env, monkeypatch, caplog):
    import logging
    prefix = "SECRET-PROMPT-" + "Q" * 64
    sess = _FakeSession(_FakeResp(status=500, text_data=prefix + "tail"))
    _use_session(monkeypatch, sess)
    with caplog.at_level(logging.WARNING, logger="proxy"):
        with pytest.raises(HTTPException):
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:zai/glm-5", 16, api_key="k",
            )
    assert prefix not in caplog.text
    assert "QQQQ" not in caplog.text


@pytest.mark.asyncio
async def test_fallback_preserves_full_vendor_model(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    with patch.object(cmd_mod, "call_command_code_direct",
                      new=AsyncMock(return_value="cli-text")) as cli:
        out = await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}],
            "command_api:evil/glm-5", 64, api_key="k",
        )
    assert out == "cli-text"
    assert cli.await_args[0][2] == "command_code:evil/glm-5"


@pytest.mark.asyncio
async def test_fallback_cli_503_one_shot(api_env, monkeypatch):
    sess = _FakeSession(_FakeResp(status=403, text_data=_upgrade_body()))
    _use_session(monkeypatch, sess)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)
    monkeypatch.setattr(proxy, "max_retries", 2, raising=False)
    cli = AsyncMock(side_effect=HTTPException(status_code=503, detail="cli down"))
    with patch.object(cmd_mod, "call_command_code_direct", new=cli):
        with pytest.raises(HTTPException) as ei:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:anyvendor/glm-5", 64, api_key="k",
            )
    assert ei.value.status_code == 503
    assert len(sess.calls) == 1
    assert cli.await_count == 1


def test_module_source_has_no_yolo():
    src = open(cmd_mod.__file__).read()
    assert "--yolo" not in src
    assert "dangerously-skip" not in src


MARK = "SENSITIVE-MARKER-9f8e7d6c5b"


def _cli_env(monkeypatch):
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", "/fake/cmd", raising=False)


@pytest.mark.asyncio
async def test_cli_prompt_via_stdin_not_argv_direct(monkeypatch):
    _cli_env(monkeypatch)
    secret = "NPC-SECRET-PII-" + "Z" * 16
    seen = {}

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        seen["args"] = list(cmd_args)
        seen["kwargs"] = kwargs
        return (b'[{"type":"text","text":"ok"}]', b"")

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        await cmd_mod.call_command_code_direct(
            secret, [{"role": "user", "content": secret}], "command_code:glm-5", 16,
        )
    argv_text = " ".join(seen["args"])
    assert secret not in argv_text
    assert seen["kwargs"]["prompt_input"] == seen["kwargs"]["expected_prompt"].encode("utf-8")
    assert secret in seen["kwargs"]["prompt_input"].decode("utf-8")
    assert seen["kwargs"]["expected_prompt"] != ""
    assert "--print" in seen["args"]


@pytest.mark.asyncio
async def test_cli_prompt_via_stdin_not_argv_stream(monkeypatch):
    _cli_env(monkeypatch)
    secret = "NPC-STREAM-PII-" + "W" * 16
    seen = {}

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        seen["args"] = list(cmd_args)
        seen["kwargs"] = kwargs
        return (b'[{"type":"text","text":"ok"}]', b"")

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        async for _ in cmd_mod.call_command_code_streaming(
            secret, [{"role": "user", "content": secret}], "command_code:glm-5", 16,
        ):
            pass
    argv_text = " ".join(seen["args"])
    assert secret not in argv_text
    assert seen["kwargs"]["prompt_input"] == seen["kwargs"]["expected_prompt"].encode("utf-8")
    assert secret in seen["kwargs"]["prompt_input"].decode("utf-8")


@pytest.mark.asyncio
async def test_run_with_interceptor_backward_compat_no_prompt_input(monkeypatch):
    import asyncio as _aio
    _cli_env(monkeypatch)
    calls = {}

    class _FakeProc:
        returncode = 0

        def kill(self):
            pass

        async def wait(self):
            return 0

        async def communicate(self):
            calls["kwargs"] = {}
            return (b"out", b"")

    async def _fake_create(*args, **kwargs):
        calls["args"] = args
        calls["create_kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(_aio, "create_subprocess_exec", _fake_create)
    out, err = await cmd_mod._run_with_interceptor(
        ["cmd", "--print"], {}, expected_prompt="hello",
    )
    assert out == b"out"
    assert "stdin" not in calls["create_kwargs"]


def test_module_source_no_prompt_in_argv():
    src = open(cmd_mod.__file__).read()
    assert '"--print", full_prompt' not in src


@pytest.mark.asyncio
async def test_direct_empty_stdout_redacted(monkeypatch, caplog):
    import logging
    _cli_env(monkeypatch)
    secret_stderr = f"NPC prompt leak {MARK} sk-ant-cred".encode()

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        return (b"", secret_stderr)

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        with caplog.at_level(logging.WARNING, logger="proxy"):
            with pytest.raises(HTTPException) as ei:
                await cmd_mod.call_command_code_direct(
                    None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
                )
    assert ei.value.status_code == 500
    assert MARK not in ei.value.detail
    assert MARK not in caplog.text


@pytest.mark.asyncio
async def test_direct_cli_exception_redacted(monkeypatch, caplog):
    import logging
    _cli_env(monkeypatch)

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        raise RuntimeError(f"boom {MARK} sk-ant-cred")

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        with caplog.at_level(logging.ERROR, logger="proxy"):
            with pytest.raises(HTTPException) as ei:
                await cmd_mod.call_command_code_direct(
                    None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
                )
    assert ei.value.status_code == 500
    assert MARK not in ei.value.detail
    assert MARK not in caplog.text


@pytest.mark.asyncio
async def test_streaming_empty_stdout_redacted(monkeypatch, caplog):
    import logging
    _cli_env(monkeypatch)
    secret_stderr = f"NPC prompt leak {MARK} sk-ant-cred".encode()

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        return (b"", secret_stderr)

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        with caplog.at_level(logging.WARNING, logger="proxy"):
            events = [c async for c in cmd_mod.call_command_code_streaming(
                None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
            )]
    assert MARK not in "".join(events)
    assert MARK not in caplog.text


@pytest.mark.asyncio
async def test_streaming_cli_exception_redacted(monkeypatch, caplog):
    import logging
    _cli_env(monkeypatch)

    async def _fake_run(cmd_args, proxy_env, **kwargs):
        raise RuntimeError(f"boom {MARK} sk-ant-cred")

    with patch.object(cmd_mod, "_run_with_interceptor", new=_fake_run):
        with caplog.at_level(logging.ERROR, logger="proxy"):
            events = [c async for c in cmd_mod.call_command_code_streaming(
                None, [{"role": "user", "content": "hi"}], "command_code:glm-5", 16,
            )]
    assert MARK not in "".join(events)
    assert MARK not in caplog.text


@pytest.mark.asyncio
async def test_interceptor_handlers_redacted(monkeypatch, caplog):
    import logging
    import aiohttp

    async def _run_case(exc):
        class _FakeSession:
            def request(self, **kwargs):
                raise exc
        req = _FakeReq({
            cmd_mod._CMD_UPSTREAM_SESSION: _FakeSession(),
            cmd_mod._CMD_UPSTREAM_TIMEOUT: aiohttp.ClientTimeout(total=1),
        })
        with caplog.at_level(logging.WARNING):
            resp = await cmd_mod._cmd_interceptor_handler(req)
        assert resp.status == 502
        assert MARK not in resp.body.decode("utf-8", errors="replace")
        return resp

    import aiohttp as _aio
    await _run_case(_aio.ClientError(f"net down {MARK} sk-ant-cred"))
    caplog.clear()
    await _run_case(RuntimeError(f"sess closed {MARK} sk-ant-cred"))
    assert MARK not in caplog.text


class _FakeReq:
    method = "GET"
    path = "/other"
    path_qs = "/other"
    headers = {}

    def __init__(self, app):
        self.app = app

    async def read(self):
        return b"{}"
