"""Integration: verified read-only credential discovery wired into routes.

Isolated: tmp HOME + PI_CODING_AGENT_DIR, monkeypatched proxy._cfg,
never reads real auth files, no live network (provider calls mocked).
"""
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy
import proxy_internal.providers.command_code as cmd_mod
from proxy_internal.providers import opencode as oc_mod

SYNTH = 9007199254740991

@pytest.fixture
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    agent = tmp_path / "agent"
    home.mkdir()
    agent.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setattr(proxy, "_cfg", {}, raising=False)
    monkeypatch.setattr(proxy, "command_code_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "opencode_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "opencode_go_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", None, raising=False)
    return home, agent

def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path

def _body(model, stream=False):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}],
            "stream": stream}

# --- command_code resolver precedence ---

def test_cmd_env_beats_config_beats_discovery(isolated, monkeypatch):
    home, agent = isolated
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "PI-KEY"}})
    proxy._cfg["command_code_api_key"] = "CFG-KEY"
    monkeypatch.setenv("COMMAND_CODE_API_KEY", "ENV-KEY")
    assert proxy._resolve_command_code_api_key() == "ENV-KEY"
    monkeypatch.delenv("COMMAND_CODE_API_KEY")
    assert proxy._resolve_command_code_api_key() == "CFG-KEY"

def test_cmd_config_beats_discovery(isolated):
    _, agent = isolated
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "PI-KEY"}})
    proxy._cfg["command_code_api_key"] = "CFG-KEY"
    assert proxy._resolve_command_code_api_key() == "CFG-KEY"

def test_cmd_pi_before_settings_before_cli(isolated):
    home, agent = isolated
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "PI-KEY"}})
    _write(agent / "settings.json", {"command-code.apiKey": "SET-KEY"})
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    assert proxy._resolve_command_code_api_key() == "PI-KEY"

def test_cmd_synthetic_oauth_then_settings_then_cli(isolated):
    home, agent = isolated
    _write(agent / "auth.json", {"command-code": {
        "type": "oauth", "access": "SYNTH", "refresh": "SYNTH",
        "expires": SYNTH}})
    assert proxy._resolve_command_code_api_key() == "SYNTH"
    _write(agent / "auth.json", {"command-code": {
        "type": "oauth", "access": "A", "refresh": "B", "expires": SYNTH}})
    _write(agent / "settings.json", {"command-code.apiKey": "SET-KEY"})
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    assert proxy._resolve_command_code_api_key() == "SET-KEY"

def test_cmd_cli_last_resort(isolated):
    home, _ = isolated
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    assert proxy._resolve_command_code_api_key() == "CLI-KEY"

def test_cmd_rotation_after_import_picked_up(isolated):
    _, agent = isolated
    assert proxy._resolve_command_code_api_key() is None
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "LATE-KEY"}})
    assert proxy._resolve_command_code_api_key() == "LATE-KEY"

def test_provider_isolation_cmd_ignores_opencode(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {
        "opencode": {"type": "api_key", "key": "WRONG"},
        "opencode-go": {"type": "api_key", "key": "WRONG-GO"}})
    assert proxy._resolve_command_code_api_key() is None

# --- opencode resolver ---

def test_opencode_env_beats_globals_beats_discovery(isolated, monkeypatch):
    _, agent = isolated
    _write(agent / "auth.json",
            {"opencode-go": {"type": "api_key", "key": "PI-GO"}})
    monkeypatch.setattr(proxy, "opencode_go_api_key", "ATTR-GO",
                        raising=False)
    monkeypatch.setattr(proxy, "opencode_api_key", "ATTR-ZEN",
                        raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "ENV-KEY")
    _, _, key, plan = oc_mod._resolve_opencode("opencode-go:kimi-k2.6")
    assert (key, plan) == ("ENV-KEY", "Go")
    monkeypatch.delenv("OPENCODE_API_KEY")
    _, _, key, _ = oc_mod._resolve_opencode("opencode-go:kimi-k2.6")
    assert key == "ATTR-GO"

def test_opencode_plan_specific_and_cross_fallback(isolated, monkeypatch):
    monkeypatch.setattr(proxy, "opencode_go_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "opencode_api_key", "ZEN-ONLY", raising=False)
    _, _, key, plan = oc_mod._resolve_opencode("opencode-go:kimi-k2.6")
    assert (key, plan) == ("ZEN-ONLY", "Go")

def test_opencode_discovery_per_plan(isolated):
    _, agent = isolated
    _write(agent / "auth.json",
            {"opencode-go": {"type": "api_key", "key": "PI-GO"}})
    _, _, go_key, _ = oc_mod._resolve_opencode("opencode-go:kimi-k2.6")
    _, _, zen_key, _ = oc_mod._resolve_opencode("opencode:big-pickle")
    assert go_key == "PI-GO"
    assert zen_key == "PI-GO"  # Zen/Go share one key
    assert oc_mod._resolve_opencode("opencode-go:kimi-k2.6")[3] == "Go"
    assert oc_mod._resolve_opencode("opencode:big-pickle")[3] == "Zen"

def test_opencode_ignores_command_code_store(isolated):
    _, agent = isolated
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "WRONG"}})
    assert oc_mod._resolve_opencode("opencode-go:kimi-k2.6")[2] is None

# --- provider direct-caller fallback ---

@pytest.mark.asyncio
async def test_command_api_key_fallback_attr_then_resolver(isolated,
                                                            monkeypatch):
    from tests.test_command_code_api import _FakeSession, _ok_resp, _use_session
    _, agent = isolated
    sess = _FakeSession(_ok_resp("via-discovery"))
    _use_session(monkeypatch, sess)
    _write(agent / "auth.json",
            {"command-code": {"type": "api_key", "key": "DISC-KEY"}})
    out = await cmd_mod.call_command_code_api_direct(
        None, [{"role": "user", "content": "hi"}],
        "command_api:zai/glm-5", 16)
    assert out == "via-discovery"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer DISC-KEY"

@pytest.mark.asyncio
async def test_command_api_missing_key_503_no_network(isolated, monkeypatch):
    from tests.test_command_code_api import _FakeSession, _ok_resp, _use_session
    sess = _FakeSession(_ok_resp())
    _use_session(monkeypatch, sess)
    with pytest.raises(HTTPException) as ei:
        await cmd_mod.call_command_code_api_direct(
            None, [{"role": "user", "content": "hi"}],
            "command_api:zai/glm-5", 16)
    assert ei.value.status_code == 503
    assert sess.calls == []

# --- route forwarding ---

class TestRouteForwarding:
    def test_direct_forwards_discovered_key(self, test_client, proxy_module,
                                            isolated):
        _, agent = isolated
        _write(agent / "auth.json",
                {"command-code": {"type": "api_key", "key": "DISC-KEY"}})
        with patch.object(proxy_module, "call_command_code_api_direct",
                            new_callable=AsyncMock,
                            return_value="api-hello") as mock_api, \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("command_api:zai/glm-5"))
        assert resp.status_code == 200
        assert mock_api.await_count == 1
        assert mock_api.call_args[1].get("api_key") == "DISC-KEY"
        assert "DISC-KEY" not in resp.text

    def test_stream_forwards_discovered_key(self, test_client, proxy_module,
                                            isolated):
        _, agent = isolated
        _write(agent / "auth.json",
                {"command-code": {"type": "api_key", "key": "DISC-KEY"}})
        seen = {}

        async def fake_stream(*a, **k):
            seen.update(k)
            yield "data: [DONE]\n\n"

        with patch.object(proxy_module, "call_command_code_api_streaming",
                            new=fake_stream), \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("command_api:zai/glm-5",
                                                stream=True))
        assert resp.status_code == 200
        assert seen.get("api_key") == "DISC-KEY"

    def test_missing_key_503_no_network(self, test_client, proxy_module,
                                        isolated):
        with patch.object(proxy_module, "call_command_code_api_direct",
                            new_callable=AsyncMock) as mock_api, \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("command_api:zai/glm-5"))
        assert resp.status_code == 503
        assert mock_api.await_count == 0

    def test_opencode_go_route_uses_discovery(self, test_client,
                                                proxy_module, isolated):
        _, agent = isolated
        _write(agent / "auth.json",
                {"opencode-go": {"type": "api_key", "key": "PI-GO"}})
        with patch.object(proxy_module, "call_opencode_direct",
                            new_callable=AsyncMock,
                            return_value="go-hi") as mock_go, \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("opencode-go:kimi-k2.6"))
        assert resp.status_code == 200
        assert mock_go.await_count == 1

    def test_opencode_missing_key_503(self, test_client, proxy_module,
                                        isolated):
        with patch.object(proxy_module, "call_opencode_direct",
                            new_callable=AsyncMock) as mock_go, \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("opencode-go:kimi-k2.6"))
        assert resp.status_code == 503
        assert mock_go.await_count == 0

    def test_resolver_called_once_per_request(self, test_client,
                                                proxy_module, isolated,
                                                monkeypatch):
        calls = []

        def rotating():
            calls.append(1)
            return "FIRST-KEY" if len(calls) == 1 else None

        monkeypatch.setattr(proxy_module, "_resolve_command_code_api_key",
                            rotating, raising=False)
        with patch.object(proxy_module, "call_command_code_api_direct",
                            new_callable=AsyncMock,
                            return_value="api-hello") as mock_api, \
                patch.object(proxy_module, "request_stats",
                            MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions",
                                    json=_body("command_api:zai/glm-5"))
        assert resp.status_code == 200
        assert len(calls) == 1
        assert mock_api.call_args[1].get("api_key") == "FIRST-KEY"

# --- resolver error semantics ---

def test_cmd_malformed_store_returns_none(isolated):
    _, agent = isolated
    bad = agent / "auth.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not-json", encoding="utf-8")
    assert proxy._resolve_command_code_api_key() is None

def test_cmd_logic_error_propagates(isolated, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(disc, "discover_command_code_key",
                        MagicMock(side_effect=RuntimeError("boom-logic")))
    monkeypatch.setattr(proxy, "_cfg", {}, raising=False)
    with pytest.raises(RuntimeError, match="boom-logic"):
        proxy._resolve_command_code_api_key()

def test_opencode_malformed_store_returns_none(isolated):
    _, agent = isolated
    bad = agent / "auth.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not-json", encoding="utf-8")
    assert oc_mod._resolve_opencode("opencode-go:kimi-k2.6")[2] is None

def test_opencode_logic_error_propagates(isolated, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(disc, "discover_opencode_key",
                        MagicMock(side_effect=RuntimeError("boom-logic")))
    monkeypatch.setattr(proxy, "opencode_go_api_key", None, raising=False)
    monkeypatch.setattr(proxy, "opencode_api_key", None, raising=False)
    with pytest.raises(RuntimeError, match="boom-logic"):
        oc_mod._resolve_opencode("opencode-go:kimi-k2.6")

def test_command_api_key_resolver_error_propagates(isolated, monkeypatch):
    monkeypatch.setattr(proxy, "_resolve_command_code_api_key",
                        MagicMock(side_effect=RuntimeError("boom-logic")),
                        raising=False)
    monkeypatch.setattr(proxy, "command_code_api_key", "stale-attr",
                        raising=False)
    with pytest.raises(RuntimeError, match="boom-logic"):
        cmd_mod._command_api_key()
