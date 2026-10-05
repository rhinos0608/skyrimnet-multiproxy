"""Shared test fixtures for proxy tests.

Strategy: import proxy with shutil.which mocked to None so no CLI auth
capture runs at startup. The FastAPI app object is available on proxy.app.
All module-level globals (auth, codex_auth, etc.) are initialised but inert.
"""
import sys, os
import pytest
from unittest.mock import patch

# Ensure proxy is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import proxy at module level with shutil.which mocked so no CLI auth capture
# runs at collection time. Test modules (e.g. test_mcp_server.py via
# proxy_internal.registry, mcp_server.py:63) import proxy at COLLECTION time,
# before any fixture runs. Module-level import ensures proxy lands in
# sys.modules with None CLI paths before any test module is collected.
with patch("shutil.which", return_value=None), patch(
    "proxy_internal.auth.antigravity.native_antigravity_enabled", return_value=False
):
    import proxy  # noqa: F401


@pytest.fixture(autouse=True)
def isolate_native_keychain(monkeypatch):
    # Legacy tests remain platform-independent. Native tests explicitly enable
    # the mode and install a synthetic Keychain reader.
    monkeypatch.setattr("proxy_internal.auth.antigravity.native_antigravity_enabled", lambda: False)


@pytest.fixture()
def proxy_module(monkeypatch, tmp_path):
    """Return the already-imported proxy module.
    Resets all provider-level globals that may leak from other tests.
    Uses monkeypatch to auto-restore values after the test.
    """
    import proxy
    # Never read the developer's saved Antigravity account during discovery tests.
    monkeypatch.setattr(proxy, "ANTIGRAVITY_AUTH_FILE", str(tmp_path / "antigravity-auth.json"))
    # Reset all provider-level state (module-level globals may leak)
    monkeypatch.setattr(proxy, "vertex_api_key", None)
    monkeypatch.setattr(proxy, "bytedance_api_key", None)
    monkeypatch.setattr(proxy, "fireworks_api_key", None)
    monkeypatch.setattr(proxy, "nvidia_api_key", None)
    monkeypatch.setattr(proxy, "kilo_api_key", None)
    monkeypatch.setattr(proxy, "openrouter_api_key", None)
    monkeypatch.setattr(proxy, "ollama_api_key", None)
    monkeypatch.setattr(proxy, "zai_api_key", None)
    monkeypatch.setattr(proxy, "xiaomi_api_key", None)
    monkeypatch.setattr(proxy, "opencode_api_key", None)
    monkeypatch.setattr(proxy, "opencode_go_api_key", None)
    # Reset CLI paths to remove machine-dependence on real installed CLIs
    monkeypatch.setattr(proxy, "CLAUDE_PATH", None)
    monkeypatch.setattr(proxy, "CODEX_PATH", None)
    monkeypatch.setattr(proxy, "COMMAND_CODE_PATH", None)
    return proxy


@pytest.fixture()
def test_client(proxy_module):
    """FastAPI TestClient that does NOT run the lifespan (no subprocess spawning).

    Instantiating TestClient without using it as a context manager skips the
    lifespan startup/shutdown events, preventing CLI auth subprocesses from running.
    """
    from fastapi.testclient import TestClient
    client = TestClient(proxy_module.app, raise_server_exceptions=True)
    client.get("/")
    yield client
