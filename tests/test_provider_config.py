"""Tests for the providers section of config.json + related registry changes."""
import pytest


@pytest.fixture()
def mcp_module(proxy_module):
    # MCP mode is intentionally not shipped in this checkout (proxy-only runtime);
    # the CLI surface for it was removed from proxy.py. Skip instead of failing.
    mcp_server = pytest.importorskip(
        "mcp_server", reason="MCP mode is not part of this checkout's runtime"
    )
    return mcp_server


class TestRegistryDefaults:
    """Defaults must match model ids that actually appear on the live proxy."""

    def test_fireworks_default_is_live_model(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "fireworks")
        assert p.default_model == "fireworks:kimi-k2p5-turbo"

    def test_nvidia_default_is_live_alias(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "nvidia")
        assert p.default_model == "nvidia:deepseek-v3.2"

    def test_opencode_zen_default_is_live_model(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "opencode_zen")
        assert p.default_model == "opencode:deepseek-v4-flash-free"

    def test_opencode_go_default_is_live_model(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "opencode_go")
        assert p.default_model == "opencode-go:glm-5.3"

    def test_antigravity_default_is_prefixed(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "antigravity")
        # Was antigravity-gemini-3-pro: Gemini 3 Pro is retired upstream and now
        # answers 200 with a deprecation notice as its content. 3-flash is the
        # live-verified replacement.
        assert p.default_model == "antigravity-gemini-3-flash"

    def test_bytedance_default_is_seed_code(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "bytedance")
        assert p.default_model == "bytedance:bytedance-seed-code"


class TestModelPrefix:
    """Every provider exposes a prefix pattern for the listing tools."""

    def test_every_provider_has_model_prefix_attribute(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        for p in REGISTRY:
            assert hasattr(p, "model_prefix"), f"{p.name} missing model_prefix"

    def test_prefix_patterns(self, proxy_module):
        from proxy_internal.registry import REGISTRY
        expected = {
            "ollama": "ollama:*",
            "codex_cli": "gpt-*|codex-*",
            "bytedance": "bytedance:*",
            "fireworks": "fireworks:*",
            "nvidia": "nvidia:*",
            "vertex": "vertex:*",
            "command_code": "command_code:*",
            "command_api": "command_api:*",
            "openrouter": "*/*",
            "kilo": "kilo:*",
            "antigravity": "antigravity-*",
            "lmstudio": "lmstudio:*",
            "zai": "zai:*",
            "xiaomi": "xiaomi:*",
            "opencode_zen": "opencode:*",
            "opencode_go": "opencode-go:*",
            "qwen_cli": "qwen:*",
            "claude": None,
        }
        for p in REGISTRY:
            assert p.model_prefix == expected[p.name], f"{p.name} prefix mismatch"


class TestProviderConfigModule:
    """Unit tests for proxy_internal/provider_config.py."""

    def test_get_setting_default_when_missing(self, monkeypatch):
        from proxy_internal import provider_config
        monkeypatch.setattr(provider_config, "_cache", {})
        assert provider_config.get_provider_setting("nvidia", "enabled", True) is True
        assert provider_config.get_provider_setting("nvidia", "priority", 1000) == 1000

    def test_get_setting_returns_configured_value(self, monkeypatch):
        from proxy_internal import provider_config
        monkeypatch.setattr(provider_config, "_cache", {"nvidia": {"enabled": False}})
        assert provider_config.get_provider_setting("nvidia", "enabled", True) is False

    def test_unknown_provider_returns_default(self, monkeypatch):
        from proxy_internal import provider_config
        monkeypatch.setattr(provider_config, "_cache", {})
        assert provider_config.get_provider_setting("made-up", "enabled", True) is True

    def test_validate_patch_clamps_priority(self):
        from proxy_internal.provider_config import validate_patch
        patch = {"priority": -5}
        assert validate_patch(patch)["priority"] == 0
        patch = {"priority": 999999}
        assert validate_patch(patch)["priority"] == 10000

    def test_validate_patch_clamps_timeout(self):
        from proxy_internal.provider_config import validate_patch
        assert validate_patch({"timeout_ms": 500})["timeout_ms"] == 1000
        assert validate_patch({"timeout_ms": 99999999})["timeout_ms"] == 3600000

    def test_validate_patch_rejects_non_int(self):
        from proxy_internal.provider_config import validate_patch
        with pytest.raises(ValueError, match="priority"):
            validate_patch({"priority": "high"})
        with pytest.raises(ValueError, match="timeout_ms"):
            validate_patch({"timeout_ms": 1.5})

    def test_validate_patch_enabled_must_be_bool(self):
        from proxy_internal.provider_config import validate_patch
        with pytest.raises(ValueError, match="enabled"):
            validate_patch({"enabled": "yes"})

    def test_validate_patch_drops_unknown_fields(self):
        from proxy_internal.provider_config import validate_patch
        out = validate_patch({"enabled": True, "bogus": 42})
        assert "bogus" not in out
        assert out["enabled"] is True

    def test_known_provider_names_matches_registry(self, proxy_module):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        # Force population
        provider_config._populate_known_names()
        assert provider_config.KNOWN_PROVIDER_NAMES == {p.name for p in REGISTRY}


class TestConfigStoreLockAndRoundtrip:
    def test_providers_block_roundtrips_through_save_load(self, tmp_path, monkeypatch):
        from proxy_internal import config_store
        monkeypatch.setattr(config_store, "CONFIG_FILE", str(tmp_path / "config.json"))
        cfg = {"providers": {"nvidia": {"enabled": False}, "claude": {"priority": 1}}}
        config_store._save_config(cfg)
        loaded = config_store._load_config()
        assert loaded["providers"] == cfg["providers"]

    def test_providers_block_is_not_encrypted(self, tmp_path, monkeypatch):
        """Provider settings are not secrets — must not go through Fernet."""
        import json
        from proxy_internal import config_store
        monkeypatch.setattr(config_store, "CONFIG_FILE", str(tmp_path / "config.json"))
        cfg = {"providers": {"nvidia": {"enabled": False}}}
        config_store._save_config(cfg)
        with open(tmp_path / "config.json") as f:
            raw = json.load(f)
        assert raw["providers"]["nvidia"]["enabled"] is False  # plaintext

    def test_kilo_api_key_is_encrypted_at_rest(self, tmp_path, monkeypatch):
        """Kilo API keys must not be written to disk in plaintext."""
        import json
        from proxy_internal import config_store
        monkeypatch.setattr(config_store, "CONFIG_FILE", str(tmp_path / "config.json"))
        cfg = {"kilo_api_key": "kilo-secret"}
        config_store._save_config(cfg)
        with open(tmp_path / "config.json") as f:
            raw = json.load(f)
        assert raw["kilo_api_key"] != "kilo-secret"

    @pytest.mark.asyncio
    async def test_config_lock_serializes_concurrent_patches(self, tmp_path, monkeypatch):
        """Two concurrent read-modify-write calls must both land; no lost update."""
        import asyncio
        from proxy_internal import config_store
        monkeypatch.setattr(config_store, "CONFIG_FILE", str(tmp_path / "config.json"))
        config_store._save_config({"providers": {}})

        async def patch(name, field, value):
            async with config_store.config_lock():
                cfg = config_store._load_config()
                cfg.setdefault("providers", {}).setdefault(name, {})[field] = value
                config_store._save_config(cfg)

        await asyncio.gather(
            patch("nvidia", "enabled", False),
            patch("claude", "priority", 1),
        )
        final = config_store._load_config()
        assert final["providers"]["nvidia"]["enabled"] is False
        assert final["providers"]["claude"]["priority"] == 1


class TestHealthFnWrapping:
    def test_enabled_false_makes_health_fn_false(self, monkeypatch):
        """With the underlying provider healthy, enabled=false still returns False."""
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "claude")
        monkeypatch.setattr(
            provider_config, "_cache", {"claude": {"enabled": False}},
        )
        # `auth.is_ready` is a read-only @property; replace the auth object
        # with a minimal stub exposing a True is_ready.
        import types
        monkeypatch.setattr("proxy.auth", types.SimpleNamespace(is_ready=True), raising=False)
        assert p.health_fn() is False

    def test_enabled_missing_preserves_underlying_health(self, monkeypatch):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "claude")
        monkeypatch.setattr(provider_config, "_cache", {})
        # `auth.is_ready` is a read-only @property; replace the auth object
        # with a minimal stub exposing a True is_ready.
        import types
        monkeypatch.setattr("proxy.auth", types.SimpleNamespace(is_ready=True), raising=False)
        assert p.health_fn() is True

    def test_enabled_true_explicit_preserves_underlying_health(self, monkeypatch):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "claude")
        monkeypatch.setattr(
            provider_config, "_cache", {"claude": {"enabled": True}},
        )
        # `auth.is_ready` is a read-only @property; replace the auth object
        # with a minimal stub exposing a True is_ready.
        import types
        monkeypatch.setattr("proxy.auth", types.SimpleNamespace(is_ready=True), raising=False)
        assert p.health_fn() is True

    def test_registry_list_order_is_not_mutated(self, proxy_module):
        """detect_provider relies on Claude (detector=None) being LAST."""
        from proxy_internal.registry import REGISTRY
        names = [p.name for p in REGISTRY]
        assert names[-1] == "claude"

    def test_wrap_health_fns_is_idempotent(self, proxy_module):
        """Calling _wrap_health_fns again must not double-wrap."""
        from proxy_internal.registry import REGISTRY, _wrap_health_fns
        before = {p.name: p.health_fn for p in REGISTRY}
        _wrap_health_fns()
        after = {p.name: p.health_fn for p in REGISTRY}
        for name in before:
            assert before[name] is after[name], f"{name} was re-wrapped"

    def test_bytedance_enabled_with_key_preserves_underlying_health(self, proxy_module, monkeypatch):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        p = next(p for p in REGISTRY if p.name == "bytedance")
        monkeypatch.setattr(provider_config, "_cache", {"bytedance": {"enabled": True}})
        monkeypatch.setattr("proxy.bytedance_api_key", "bd-test-key", raising=False)
        assert p.health_fn() is True


class TestPrioritySort:
    def test_healthy_chat_providers_respects_priority(self, mcp_module, monkeypatch):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        for p in REGISTRY:
            monkeypatch.setattr(p, "health_fn", lambda: True)
        monkeypatch.setattr(
            provider_config, "_cache",
            {"claude": {"priority": 1}, "nvidia": {"priority": 2}},
        )
        names = [p.name for p in mcp_module.get_healthy_chat_providers()]
        # claude (priority 1) first, nvidia (priority 2) next, rest tied at 1000
        assert names[0] == "claude"
        assert names[1] == "nvidia"

    def test_healthy_providers_default_priority_preserves_registry_order(
        self, mcp_module, monkeypatch
    ):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        for p in REGISTRY:
            monkeypatch.setattr(p, "health_fn", lambda: True)
        monkeypatch.setattr(provider_config, "_cache", {})
        chat_names_in_registry = [p.name for p in REGISTRY if p.category == "chat"]
        chat_names_from_getter = [p.name for p in mcp_module.get_healthy_chat_providers()]
        assert chat_names_from_getter == chat_names_in_registry

    def test_registry_order_unchanged_after_getter(self, mcp_module, monkeypatch):
        from proxy_internal import provider_config
        from proxy_internal.registry import REGISTRY
        for p in REGISTRY:
            monkeypatch.setattr(p, "health_fn", lambda: True)
        monkeypatch.setattr(
            provider_config, "_cache", {"nvidia": {"priority": 0}},
        )
        _ = mcp_module.get_healthy_chat_providers()
        names_after = [p.name for p in REGISTRY]
        assert names_after[-1] == "claude", "REGISTRY order must not mutate"


class TestStartupReload:
    def test_reload_from_config_reads_providers_block(self, tmp_path, monkeypatch):
        """reload_from_config populates _cache from config.json's providers block."""
        import json
        from proxy_internal import config_store, provider_config

        monkeypatch.setattr(config_store, "CONFIG_FILE", str(tmp_path / "config.json"))
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"providers": {"nvidia": {"enabled": False}}}, f)

        monkeypatch.setattr(provider_config, "_cache", {})
        provider_config.reload_from_config()
        assert provider_config.get_provider_setting("nvidia", "enabled", True) is False
