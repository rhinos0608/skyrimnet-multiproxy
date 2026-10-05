"""Focused mapping tests: OpenRouter dash-alias slugs + OpenCode Go hint currency."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _make_mock_response(status=200, json_data=None):
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_session(resp):
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize("alias, canonical", [
    ("anthropic/claude-3-5-sonnet", "anthropic/claude-3.5-sonnet"),
    ("anthropic/claude-3-7-sonnet", "anthropic/claude-3.7-sonnet"),
])
async def test_openrouter_dash_alias_resolves_direct(proxy_module, alias, canonical):
    from proxy_internal.providers import openrouter as or_mod
    assert or_mod._resolve_openrouter_model(alias) == canonical
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "openrouter_api_key", "sk-or"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        out = await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], alias, 100,
        )
    assert out == "hi"
    assert session.post.call_args[1]["json"]["model"] == canonical


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-opus-5.5",
    "openai/gpt-6.1-sol",
])
async def test_openrouter_canonical_passthrough_direct(proxy_module, model):
    from proxy_internal.providers import openrouter as or_mod
    assert or_mod._resolve_openrouter_model(model) == model
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    session = _make_mock_session(resp)
    with patch.object(proxy_module, "openrouter_api_key", "sk-or"), \
         patch.object(proxy_module, "third_party_session", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session):
        await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], model, 100,
        )
    assert session.post.call_args[1]["json"]["model"] == model


@pytest.mark.asyncio
async def test_openrouter_dash_alias_resolves_streaming(proxy_module):
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
    open_stream = AsyncMock(return_value=(mock_resp, mock_resp))
    with patch.object(proxy_module, "openrouter_api_key", "sk-or"), \
         patch.object(proxy_module, "_open_stream_with_retry", open_stream):
        chunks = [c async for c in proxy_module.call_openrouter_streaming(
            None, [{"role": "user", "content": "hi"}],
            "anthropic/claude-3-5-sonnet", 100,
        )]
    assert chunks
    payload = open_stream.call_args.kwargs.get("json_payload") or open_stream.call_args[0][2]
    assert payload["model"] == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_go_non_chat_error_hint_names_current_chat_models(proxy_module):
    with patch.object(proxy_module, "opencode_api_key", "sk-zen"), \
         patch.object(proxy_module, "opencode_go_api_key", "sk-go"):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_opencode_direct(
                None, [{"role": "user", "content": "hi"}],
                "opencode-go:gpt-6-luna", 100,
            )
    assert exc.value.status_code == 400
    assert "kimi-k3" in exc.value.detail
    assert "mimo-v2.6-flash" in exc.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "opencode-go:mimo-v2.6-flash",
    "opencode-go:kimi-k3",
    "opencode-go:glm-5.3-flash",
    "opencode-go:deepseek-v4-flash",
])
async def test_go_allows_current_chat_models(proxy_module, model):
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
