"""Config, discovery, registry, and health wiring for command_api: + OpenCode Go.

Uses a tmp config.json and dummy key material only — no live upstream,
no real secrets, nothing written to the repo config.json.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


DUMMY_KEY = "test-command-code-key-000"
API_MODEL_ID = "command_api:deepseek/deepseek-v4-flash"


@pytest.fixture()
def tmp_config(monkeypatch, tmp_path):
    """Point config_store at a throwaway config.json."""
    from proxy_internal import config_store

    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(config_store, "CONFIG_FILE", str(cfg_file))
    return cfg_file


@pytest.fixture()
def isolated_cfg(monkeypatch, proxy_module):
    """Give proxy a fresh in-memory _cfg + no command key (auto-restored)."""
    monkeypatch.setattr(proxy_module, "_cfg", {})
    monkeypatch.setattr(proxy_module, "command_code_api_key", None, raising=False)
    return proxy_module


def _only(names, monkeypatch):
    """Disable health for every provider except `names`."""
    from proxy_internal.registry import REGISTRY

    for p in REGISTRY:
        if p.name in names:
            continue
        monkeypatch.setattr(p, "health_fn", lambda: False)


class TestCommandCodeKeyEndpoint:
    def test_set_key_saved_to_tmp_config(self, test_client, proxy_module, tmp_config, isolated_cfg):
        resp = test_client.post("/config/command-code-key", json={"key": DUMMY_KEY})
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        assert proxy_module.command_code_api_key == DUMMY_KEY
        assert isolated_cfg._cfg.get("command_code_api_key") == DUMMY_KEY
        with open(tmp_config) as f:
            raw = json.load(f)
        assert raw["command_code_api_key"] != DUMMY_KEY  # encrypted at rest
        import os as _os
        assert _os.path.realpath(str(tmp_config)) != _os.path.realpath("config.json")

    def test_set_key_encrypts_at_rest(self, test_client, proxy_module, tmp_config, isolated_cfg):
        test_client.post("/config/command-code-key", json={"key": DUMMY_KEY})
        with open(tmp_config, encoding="utf-8") as f:
            body = f.read()
        assert DUMMY_KEY not in body

    def test_clear_key(self, test_client, proxy_module, tmp_config, isolated_cfg, monkeypatch, tmp_path):
        # Isolate host auth so Pi/CLI discovery can't repopulate the key after clear.
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
        test_client.post("/config/command-code-key", json={"key": DUMMY_KEY})
        resp = test_client.post("/config/command-code-key", json={"key": ""})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert proxy_module.command_code_api_key is None
        assert "command_code_api_key" not in isolated_cfg._cfg

    def test_rejects_non_string_key(self, test_client, proxy_module, tmp_config, isolated_cfg):
        resp = test_client.post("/config/command-code-key", json={"key": 123})
        assert resp.status_code == 400

    def test_env_key_takes_precedence(self, test_client, proxy_module, tmp_config, isolated_cfg, monkeypatch):
        monkeypatch.setenv("COMMAND_CODE_API_KEY", "test-env-key-111")
        resp = test_client.post("/config/command-code-key", json={"key": DUMMY_KEY})
        assert resp.status_code == 200
        # Env wins over the just-saved config value in memory.
        assert proxy_module.command_code_api_key == "test-env-key-111"
        # Clearing config does not drop the env-provided key.
        test_client.post("/config/command-code-key", json={"key": ""})
        assert proxy_module.command_code_api_key == "test-env-key-111"

    def test_never_logs_key_value(self, test_client, proxy_module, tmp_config, isolated_cfg, caplog):
        logger = logging.getLogger("proxy")
        with caplog.at_level(logging.INFO, logger="proxy"):
            test_client.post("/config/command-code-key", json={"key": DUMMY_KEY})
        assert DUMMY_KEY not in caplog.text
        assert logger is not None


class TestCommandApiHealth:
    def test_configured_only_when_key_available(self, test_client, proxy_module, monkeypatch, tmp_path):
        # Isolate host auth: no env key, tmp HOME + Pi dir with no credentials.
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
        monkeypatch.setattr(proxy_module, "_cfg", {})
        monkeypatch.setattr(proxy_module, "command_code_api_key", None, raising=False)
        assert test_client.get("/health").json()["command_api_configured"] is False
        monkeypatch.setattr(
            proxy_module, "_resolve_command_code_api_key", lambda: DUMMY_KEY, raising=False
        )
        assert test_client.get("/health").json()["command_api_configured"] is True


class TestCommandApiLateLogin:
    def test_pi_file_login_picked_up_without_restart(
        self, test_client, proxy_module, monkeypatch, tmp_path
    ):
        """Credential written to Pi auth file after import is resolved dynamically."""
        import json as _json

        home = tmp_path / "home"
        agent = tmp_path / "agent"
        home.mkdir()
        agent.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "_cfg", {})
        monkeypatch.setattr(proxy_module, "command_code_api_key", None, raising=False)
        assert proxy_module._resolve_command_code_api_key() is None
        assert test_client.get("/health").json()["command_api_configured"] is False
        (agent / "auth.json").write_text(
            _json.dumps({"command-code": {"type": "api_key", "key": "LATE-KEY"}})
        )
        assert proxy_module._resolve_command_code_api_key() == "LATE-KEY"
        assert test_client.get("/health").json()["command_api_configured"] is True
        _only({"command_api", "command_code"}, monkeypatch)
        monkeypatch.setattr(proxy_module, "COMMAND_CODE_PATH", None)
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert API_MODEL_ID in ids


class TestCommandApiModels:
    def test_lists_api_model_when_key_configured(
        self, test_client, proxy_module, monkeypatch
    ):
        monkeypatch.setattr(
            proxy_module, "_resolve_command_code_api_key", lambda: DUMMY_KEY, raising=False
        )
        monkeypatch.setattr(proxy_module, "command_code_api_key", DUMMY_KEY, raising=False)
        monkeypatch.setattr(proxy_module, "COMMAND_CODE_PATH", None)
        _only({"command_api", "command_code"}, monkeypatch)
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert API_MODEL_ID in ids
        assert not [i for i in ids if i.startswith("command_code:")]

    def test_no_api_model_without_key(self, test_client, proxy_module, monkeypatch, tmp_path):
        # Isolate host auth so a real Pi/CLI login on this machine can't leak in.
        monkeypatch.delenv("COMMAND_CODE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
        monkeypatch.setattr(proxy_module, "_cfg", {})
        monkeypatch.setattr(
            proxy_module, "_resolve_command_code_api_key", lambda: None, raising=False
        )
        monkeypatch.setattr(proxy_module, "command_code_api_key", None, raising=False)
        monkeypatch.setattr(proxy_module, "COMMAND_CODE_PATH", None)
        _only({"command_api", "command_code"}, monkeypatch)
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert API_MODEL_ID not in ids


class TestRegistryConsistency:
    def test_command_api_entry(self, proxy_module):
        from proxy_internal.registry import REGISTRY, detect_provider

        names = [p.name for p in REGISTRY]
        assert "command_api" in names
        assert "command_code" in names  # legacy CLI provider kept
        assert names[-1] == "claude"  # fallback stays last
        p = next(p for p in REGISTRY if p.name == "command_api")
        assert p.default_model == API_MODEL_ID
        assert p.model_prefix == "command_api:*"
        assert detect_provider("command_api:zai/glm-5").name == "command_api"
        assert detect_provider("command_code:glm-5").name == "command_code"

    def test_legacy_command_code_points_at_cli(self, proxy_module):
        from proxy_internal.registry import REGISTRY

        p = next(p for p in REGISTRY if p.name == "command_code")
        assert "API key" not in p.error_reason
        assert "install" in p.error_reason.lower()
        assert "CLI" in p.description or "cli" in p.description.lower()


class TestOpenCodeGoModels:
    def _go_only(self, test_client, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module, "opencode_go_api_key", "test-go-key", raising=False)
        monkeypatch.setattr(proxy_module, "opencode_api_key", None, raising=False)
        _only({"opencode_go"}, monkeypatch)

    def test_filters_non_chat_discovered_ids(self, test_client, proxy_module, monkeypatch):
        self._go_only(test_client, proxy_module, monkeypatch)
        fake = MagicMock()
        fake.returncode = 0
        fake.stdout = (
            "opencode-go/glm-5.3\nopencode-go/kimi-k2.6\n"
            "opencode-go/minimax-m2.5\nopencode-go/gpt-6-luna\n"
            "opencode-go/mimo-v2.5\n"
        )
        monkeypatch.setattr("subprocess.run", lambda *a, **k: fake)
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert "opencode-go:glm-5.3" in ids
        assert "opencode-go:kimi-k2.6" in ids
        assert "opencode-go:mimo-v2.5" in ids
        assert "opencode-go:minimax-m2.5" not in ids
        assert "opencode-go:gpt-6-luna" not in ids

    def test_fallback_uses_current_chat_ids(self, test_client, proxy_module, monkeypatch):
        self._go_only(test_client, proxy_module, monkeypatch)

        def _boom(*a, **k):
            raise OSError("no cli")

        monkeypatch.setattr("subprocess.run", _boom)
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert "opencode-go:glm-5.3" in ids
        assert "opencode-go:kimi-k3" in ids
        assert "opencode-go:mimo-v2.6-flash" in ids
        assert "opencode-go:mimo-v2.5-pro" in ids
        assert "opencode-go:glm-5.1" not in ids
        assert "opencode-go:kimi-k2.5" not in ids
        assert "opencode-go:kimi-k2.6" not in ids
        assert not [i for i in ids if "minimax" in i]
