"""E2E routing tests for command_api: + opencode-go session header.

Local TestClient/ASGI only — no live network, no real credentials.
Provider call functions are mocked; asserts cover routing, model IDs,
safe headers, and session forwarding.
"""
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_MODEL = "command_api:zai/glm-5"
GO_MODEL = "opencode-go:kimi-k2.6"
LEGACY_MODEL = "command_code:glm-5"


def _body(model, stream=False):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


async def _fake_api_stream(*args, **kwargs):
    yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
    yield "data: [DONE]\n\n"


async def _fake_go_stream(*args, **kwargs):
    yield "data: [DONE]\n\n"


class TestCommandApiDirect:
    def test_direct_routes_to_api(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "command_code_api_key", "k-test", raising=False)
        with patch.object(proxy_module, "call_command_code_api_direct",
                          new_callable=AsyncMock, return_value="api-hello") as mock_api, \
             patch.object(proxy_module, "call_command_code_direct",
                          new_callable=AsyncMock) as mock_legacy, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json=_body(API_MODEL))
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == API_MODEL
        assert data["choices"][0]["message"]["content"] == "api-hello"
        assert mock_api.await_count == 1
        assert mock_legacy.await_count == 0
        call_kwargs = mock_api.call_args
        assert call_kwargs[0][2] == API_MODEL  # model id preserved
        assert "k-test" not in resp.text  # key never leaks

    def test_direct_case_insensitive(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "command_code_api_key", "k-test", raising=False)
        with patch.object(proxy_module, "call_command_code_api_direct",
                          new_callable=AsyncMock, return_value="x") as mock_api, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json=_body("COMMAND_API:zai/glm-5"))
        assert resp.status_code == 200
        assert mock_api.await_count == 1

    def test_stream_routes_to_api(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "command_code_api_key", "k-test", raising=False)
        with patch.object(proxy_module, "call_command_code_api_streaming",
                          new=_fake_api_stream), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json=_body(API_MODEL, stream=True))
        assert resp.status_code == 200
        assert "data:" in resp.text
        assert "[DONE]" in resp.text

    def test_missing_key_503(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "command_code_api_key", None, raising=False)
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "_resolve_command_code_api_key", lambda: None, raising=False)
        with patch.object(proxy_module, "call_command_code_api_direct",
                          new_callable=AsyncMock) as mock_api:
            resp = test_client.post("/v1/chat/completions", json=_body(API_MODEL))
        assert resp.status_code == 503
        assert "COMMAND_CODE_API_KEY" in resp.json()["detail"]
        assert mock_api.await_count == 0


class TestLegacyUnaffected:
    def test_legacy_cli_route(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "COMMAND_CODE_PATH", "/usr/bin/cmd", raising=False)
        with patch.object(proxy_module, "call_command_code_direct",
                          new_callable=AsyncMock, return_value="cli-hello") as mock_legacy, \
             patch.object(proxy_module, "call_command_code_api_direct",
                          new_callable=AsyncMock) as mock_api, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json=_body(LEGACY_MODEL))
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "cli-hello"
        assert mock_legacy.await_count == 1
        assert mock_api.await_count == 0


class TestGoSessionHeader:
    def test_go_direct_forwards_session(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "opencode_go_api_key", "sk-go", raising=False)
        with patch.object(proxy_module, "call_opencode_direct",
                          new_callable=AsyncMock, return_value="go-hi") as mock_go, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/chat/completions", json=_body(GO_MODEL),
                headers={"x-opencode-session": "sess-abc"},
            )
        assert resp.status_code == 200
        assert mock_go.await_count == 1
        assert mock_go.call_args[1].get("opencode_session_id") == "sess-abc"

    def test_go_direct_absent_no_session_kwarg(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "opencode_go_api_key", "sk-go", raising=False)
        with patch.object(proxy_module, "call_opencode_direct",
                          new_callable=AsyncMock, return_value="go-hi") as mock_go, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json=_body(GO_MODEL))
        assert resp.status_code == 200
        assert "opencode_session_id" not in mock_go.call_args[1]

    def test_go_stream_forwards_session(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "opencode_go_api_key", "sk-go", raising=False)
        seen = {}

        async def fake_stream(*args, **kwargs):
            seen.update(kwargs)
            yield "data: [DONE]\n\n"

        with patch.object(proxy_module, "call_opencode_streaming", new=fake_stream), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/chat/completions", json=_body(GO_MODEL, stream=True),
                headers={"x-opencode-session": "sess-xyz"},
            )
        assert resp.status_code == 200
        assert seen.get("opencode_session_id") == "sess-xyz"

    def test_wrong_provider_header_not_forwarded(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "command_code_api_key", "k-test", raising=False)
        with patch.object(proxy_module, "call_command_code_api_direct",
                          new_callable=AsyncMock, return_value="api-hi") as mock_api, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/chat/completions", json=_body(API_MODEL),
                headers={"x-opencode-session": "sess-evil"},
            )
        assert resp.status_code == 200
        assert "opencode_session_id" not in mock_api.call_args[1]
