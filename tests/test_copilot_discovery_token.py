"""Upstream error scrubbing + 400 debug-log redaction (no network)."""
import pytest


def test_scrub_helpers_bound_and_single_line():
    from proxy_internal.providers.command_code import _scrub_upstream_reason
    from proxy_internal.providers.opencode import _scrub_opencode_reason

    messy = "bad request\nsecond line\r\nthird   spaced " + "z" * 500
    for fn in (_scrub_upstream_reason, _scrub_opencode_reason):
        out = fn(messy)
        assert len(out) <= 300
        assert "\n" not in out and "\r" not in out


def test_scrub_helpers_redact_key_shaped_values():
    from proxy_internal.providers.command_code import _scrub_upstream_reason
    from proxy_internal.providers.opencode import _scrub_opencode_reason

    body = "upstream said no: sk-secret-LEAK-AT-START-12345 and ghu_faketoken999 tail"
    for fn in (_scrub_upstream_reason, _scrub_opencode_reason):
        out = fn(body)
        assert "sk-secret-LEAK-AT-START-12345" not in out
        assert "ghu_faketoken999" not in out


@pytest.mark.asyncio
async def test_command_api_400_reason_in_debug_log_not_detail(monkeypatch, caplog):
    import logging
    import json as _json
    from unittest.mock import AsyncMock

    import proxy
    import proxy_internal.providers.command_code as cmd_mod

    body = _json.dumps({"error": {"message": "unknown model xyz", "code": "bad_model"}})

    class _Resp:
        status = 400

        async def text(self):
            return body

    class _CM:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Sess:
        def post(self, *a, **k):
            return _CM()

    monkeypatch.setattr(proxy, "third_party_session", _Sess(), raising=False)
    monkeypatch.setattr(proxy.auth, "session", None, raising=False)
    with caplog.at_level(logging.DEBUG, logger="proxy"):
        with pytest.raises(Exception) as exc:
            await cmd_mod.call_command_code_api_direct(
                None, [{"role": "user", "content": "hi"}],
                "command_api:zai/glm-5", 16, api_key="k-test",
            )
    assert getattr(exc.value, "status_code", None) == 400
    assert exc.value.detail == "Command Code API error (400)"
    assert "unknown model xyz" in caplog.text


@pytest.mark.asyncio
async def test_opencode_go_400_reason_in_debug_log_not_detail(monkeypatch, caplog):
    import logging
    from unittest.mock import AsyncMock, MagicMock, patch

    import proxy

    body = '{"error": "model glm-bad not found"}'
    mock_resp = AsyncMock()
    mock_resp.status = 400
    mock_resp.text = AsyncMock(return_value=body)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=mock_resp)
    session.close = AsyncMock()
    with patch.object(proxy, "opencode_api_key", "sk-zen"), \
         patch.object(proxy, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy, "third_party_session", None), \
         patch.object(proxy, "auth", MagicMock(session=None)), \
         patch.object(proxy, "create_session", return_value=session):
        with caplog.at_level(logging.DEBUG, logger="proxy"), \
             pytest.raises(Exception) as exc:
            await proxy.call_opencode_direct(
                None, [{"role": "user", "content": "hi"}],
                "opencode-go:glm-5.3", 100,
            )
    assert getattr(exc.value, "status_code", None) == 400
    assert exc.value.detail == "OpenCode Go request failed (status 400)"
    assert "model glm-bad not found" in caplog.text
