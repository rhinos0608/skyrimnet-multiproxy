"""Contract tests for OpenCode Go provider fix (TDD)."""
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _make_mock_response(status=200, json_data=None, text_data=""):
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_session(resp):
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_go_direct_sends_honest_ua_and_session(proxy_module):
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        await proxy_module.call_opencode_direct(
            None, [{"role": "user", "content": "hi"}],
            "opencode-go:kimi-k2.6", 100, opencode_session_id="sess-123",
        )
    headers = session.post.call_args[1]["headers"]
    payload = session.post.call_args[1]["json"]
    assert "opencode/" not in headers["User-Agent"].lower() or "cli" not in headers["User-Agent"].lower()
    assert "skyrimnet" in headers["User-Agent"].lower() or "proxy" in headers["User-Agent"].lower()
    assert headers["x-opencode-session"] == "sess-123"
    assert "opencode_session_id" not in payload
    assert "x-opencode-client" not in headers
    assert "x-opencode-project" not in headers


@pytest.mark.asyncio
async def test_go_direct_fallback_session_uuid(proxy_module):
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        await proxy_module.call_opencode_direct(
            None, [{"role": "user", "content": "hi"}], "opencode-go:glm-5.3", 100,
        )
    headers = session.post.call_args[1]["headers"]
    uuid_re = re.compile(r"^[0-9a-fA-F-]{8,}$")
    assert uuid_re.match(headers["x-opencode-session"])
    payload = session.post.call_args[1]["json"]
    assert "opencode_session_id" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "opencode-go:minimax-m2.7",
    "opencode-go:qwen3.8",
    "opencode-go:muse-spark-1.3",
    "opencode-go:gpt-6-luna",
])
async def test_go_direct_rejects_non_chat_family(proxy_module, model):
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_opencode_direct(
                None, [{"role": "user", "content": "hi"}], model, 100,
            )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    # Exact documented /responses + /messages IDs per https://opencode.ai/docs/go/
    "opencode-go:xai/grok-4.7",
    "opencode-go:grok-4.7",
    "opencode-go:grok-4.6",
    "opencode-go:gpt-5.6-luna",
    "opencode-go:qwen3.7-plus",
    "opencode-go:minimax-m3",
    "opencode-go:qwen3.8-max",
    "opencode-go:qwen3.8-flash",
    "opencode-go:muse-spark-1.2",
])
async def test_go_direct_rejects_documented_non_chat_ids(proxy_module, model):
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_opencode_direct(
                None, [{"role": "user", "content": "hi"}], model, 100,
            )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "opencode-go:glm-5.3",
    "opencode-go:glm-5.3-flash",
    "opencode-go:kimi-k2.6",
    "opencode-go:kimi-k2.7-code",
    "opencode-go:kimi-k3",
])
async def test_go_direct_allows_chat_models_no_false_positive(proxy_module, model):
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        out = await proxy_module.call_opencode_direct(
            None, [{"role": "user", "content": "hi"}], model, 100,
        )
    assert out == "hi"


@pytest.mark.asyncio
async def test_go_direct_error_sanitizes_secret_body(proxy_module, caplog):
    import logging
    secret = "sk-secret-LEAK-AT-START-12345" + "x" * 20
    resp = _make_mock_response(500, {}, secret + " rest of upstream body")
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        with caplog.at_level(logging.ERROR, logger="proxy_internal.providers.opencode"), \
             pytest.raises(HTTPException) as exc:
            await proxy_module.call_opencode_direct(
                None, [{"role": "user", "content": "hi"}],
                "opencode-go:glm-5.3", 100,
            )
    assert secret not in str(exc.value.detail)
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["opencode-go:glm-5.3", "opencode-go:kimi-k2.6"])
async def test_go_streaming_supports_chat_model(proxy_module, model):
    from proxy_internal.providers import opencode as oc_mod
    sse = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    content = MagicMock()

    async def _iter(chunks):
        for c in chunks:
            yield c
    content.iter_any.return_value = _iter([sse])
    mock_resp.content = content
    session = _make_mock_session(mock_resp)
    open_stream = AsyncMock(return_value=(mock_resp, mock_resp))
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session), \
         patch.object(proxy_module, "_open_stream_with_retry", open_stream):
        chunks = [c async for c in proxy_module.call_opencode_streaming(
            None, [{"role": "user", "content": "hi"}], model, 100,
            opencode_session_id="sess-abc",
        )]
    assert chunks
    headers = open_stream.call_args.kwargs["headers"]
    assert headers["x-opencode-session"] == "sess-abc"
    payload = open_stream.call_args.kwargs.get("json_payload") or open_stream.call_args[0][2]
    assert "opencode_session_id" not in payload


@pytest.mark.asyncio
async def test_zen_preserved(proxy_module):
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", None), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        await proxy_module.call_opencode_direct(
            None, [{"role": "user", "content": "hi"}], "opencode:big-pickle", 100,
        )
    headers = session.post.call_args[1]["headers"]
    assert headers["User-Agent"] == "opencode/latest/1.3.15/cli"
    assert headers["x-opencode-client"] == "cli"
