"""Wiring tests: on-disk credential auto-discovery into the runtime.

Isolated: tmp HOME + PI_CODING_AGENT_DIR, in-memory _cfg, cleared env.
Never touches real ~/.pi/agent/auth.json or any live provider.
"""
import json
import logging

import pytest

FRESH = 9999999999999
STALE = 1000

OR_KEY = "or-disk-key-001"
OL_KEY = "ol-disk-key-002"
VX_KEY = "vx-disk-key-003"
CLAUDE_TOK = "claude-disk-tok-005"
CODEX_TOK = "codex-disk-tok-006"
ANTI_TOK = "anti-disk-tok-008"

@pytest.fixture()
def iso(monkeypatch, tmp_path, proxy_module):
    home = tmp_path / "home"
    agent = tmp_path / "agent"
    home.mkdir()
    agent.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    for env in (
        "OPENROUTER_API_KEY", "OLLAMA_API_KEY", "VERTEX_API_KEY",
        "COMMAND_CODE_API_KEY", "OPENCODE_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(proxy_module, "_cfg", {})
    for name in ("openrouter_api_key", "ollama_api_key", "vertex_api_key"):
        monkeypatch.setattr(proxy_module, name, None, raising=False)
    proxy_module._disk_source_logged.clear()
    # Reset OAuth caches (auto-restored by monkeypatch).
    monkeypatch.setattr(proxy_module.auth, "headers", None)
    monkeypatch.setattr(proxy_module.auth, "body_template", None)
    for attr in ("access_token", "refresh_token", "account_id", "expires_at"):
        monkeypatch.setattr(proxy_module.codex_auth, attr, None)
    monkeypatch.setattr(proxy_module.antigravity_auth, "accounts", [])
    from proxy_internal.auth.antigravity import AntigravityAccount
    monkeypatch.setattr(
        proxy_module.antigravity_auth, "_legacy_account", AntigravityAccount()
    )
    return proxy_module, agent

def _write_auth(agent, blob):
    (agent / "auth.json").write_text(json.dumps(blob), encoding="utf-8")

def _oauth(access, expires=FRESH, **extra):
    rec = {"type": "oauth", "access": access, "refresh": "REF",
            "expires": expires}
    rec.update(extra)
    return rec

def _key_providers():
    return [
        ("openrouter", "openrouter", OR_KEY, "_resolve_openrouter_api_key",
        "openrouter_key_source"),
        ("ollama", "ollama", OL_KEY, "_resolve_ollama_api_key",
        "ollama_key_source"),
        ("vertex", "google-vertex", VX_KEY, "_resolve_vertex_api_key",
        "vertex_key_source"),
    ]

@pytest.mark.parametrize("pname,pid,key,resolver,sourcefn", _key_providers())
def test_key_disk_and_precedence(iso, monkeypatch, pname, pid, key,
                                resolver, sourcefn):
    proxy, agent = iso
    resolve = getattr(proxy, resolver)
    source = getattr(proxy, sourcefn)
    assert resolve() is None
    assert source() is None
    _write_auth(agent, {pid: {"type": "api_key", "key": key}})
    assert resolve() == key
    assert source() == "disk"
    # Config still wins over disk.
    monkeypatch.setattr(proxy, "_cfg", {f"{pname}_api_key": f"{pname}-cfg"})
    assert resolve() == f"{pname}-cfg"
    assert source() == "config"
    # Env still wins over both.
    monkeypatch.setenv(f"{pname.upper()}_API_KEY", f"{pname}-env")
    assert resolve() == f"{pname}-env"
    assert source() == "env"

@pytest.mark.parametrize("pname,pid,key,resolver,sourcefn", _key_providers())
def test_key_missing_malformed_not_ready(iso, pname, pid, key, resolver,
                                        sourcefn):
    proxy, agent = iso
    resolve = getattr(proxy, resolver)
    source = getattr(proxy, sourcefn)
    assert resolve() is None  # missing file
    (agent / "auth.json").write_text("not json", encoding="utf-8")
    assert resolve() is None  # malformed
    assert source() is None
    _write_auth(agent, {pid: {"type": "oauth", "access": key,
                                "refresh": "R", "expires": FRESH}})
    assert resolve() is None  # wrong credential class: no key from OAuth
    assert source() is None
    _write_auth(agent, {pid: {"type": "api_key", "key": "!evil"}})
    assert resolve() is None  # shell expression never executed

def test_key_cross_provider_isolation(iso):
    proxy, agent = iso
    _write_auth(agent, {"google-vertex": {"type": "api_key", "key": VX_KEY}})
    assert proxy._resolve_openrouter_api_key() is None
    assert proxy._resolve_ollama_api_key() is None
    assert proxy._resolve_vertex_api_key() == VX_KEY

def test_key_late_pickup_without_restart(iso):
    proxy, agent = iso
    assert proxy._resolve_vertex_api_key() is None
    _write_auth(agent, {"google-vertex": {"type": "api_key", "key": VX_KEY}})
    assert proxy._resolve_vertex_api_key() == VX_KEY
    assert proxy.vertex_api_key == VX_KEY  # mirrored for providers/*
    assert proxy.vertex_key_source() == "disk"

def test_claude_disk_ready_and_source(iso):
    proxy, agent = iso
    assert proxy.is_claude_auth_ready() is False
    assert proxy.claude_auth_source() is None
    _write_auth(agent, {"anthropic": _oauth(CLAUDE_TOK)})
    assert proxy.is_claude_auth_ready() is True
    assert proxy.claude_auth_source() == "disk"
    # MITM headers still report config when present.
    proxy.auth.headers = {"x": "y"}
    proxy.auth.body_template = {}
    assert proxy.claude_auth_source() == "config"

def test_claude_expired_malformed_not_ready(iso):
    proxy, agent = iso
    _write_auth(agent, {"anthropic": _oauth(CLAUDE_TOK, expires=STALE)})
    assert proxy.is_claude_auth_ready() is False
    assert proxy.claude_auth_source() is None
    _write_auth(agent, {"anthropic": {"type": "api_key", "key": "K"}})
    assert proxy.is_claude_auth_ready() is False  # no OAuth->key conversion
    (agent / "auth.json").write_text("bad", encoding="utf-8")
    assert proxy.is_claude_auth_ready() is False

def test_claude_isolation(iso):
    proxy, agent = iso
    _write_auth(agent, {"openai-codex": _oauth(CODEX_TOK)})
    assert proxy.is_claude_auth_ready() is False
    assert proxy.ensure_antigravity_discovery() is False

def test_codex_disk_populates_cache(iso):
    proxy, agent = iso
    assert proxy.ensure_codex_discovery() is False
    _write_auth(agent, {"openai-codex": _oauth(CODEX_TOK, accountId="ACC-1")})
    assert proxy.ensure_codex_discovery() is True
    assert proxy.codex_auth.access_token == CODEX_TOK
    assert proxy.codex_auth.account_id == "ACC-1"
    assert proxy.codex_auth_source() == "disk"
    # Late pickup after first read.
    proxy2_reset = proxy.codex_auth
    proxy2_reset.access_token = None
    proxy2_reset.expires_at = None
    assert proxy.ensure_codex_discovery() is True

def test_codex_expired_not_ready(iso):
    proxy, agent = iso
    _write_auth(agent, {"openai-codex": _oauth(CODEX_TOK, expires=STALE)})
    assert proxy.ensure_codex_discovery() is False
    assert proxy.codex_auth.access_token is None
    assert proxy.codex_auth_source() is None

def test_antigravity_disk_account(iso):
    proxy, agent = iso
    assert proxy.ensure_antigravity_discovery() is False
    _write_auth(agent, {"antigravity": _oauth(ANTI_TOK)})
    assert proxy.ensure_antigravity_discovery() is True
    assert proxy.antigravity_auth.is_ready is True
    assert proxy.antigravity_auth_source() == "disk"
    assert proxy.ensure_antigravity_discovery() is True  # idempotent

def test_antigravity_expired_not_ready(iso):
    proxy, agent = iso
    _write_auth(agent, {"antigravity": _oauth(ANTI_TOK, expires=STALE)})
    assert proxy.ensure_antigravity_discovery() is False
    assert proxy.antigravity_auth_source() is None
    assert proxy.antigravity_auth.accounts == []

def test_no_credential_value_in_logs(iso, caplog):
    proxy, agent = iso
    secrets = [OR_KEY, OL_KEY, VX_KEY, CLAUDE_TOK, CODEX_TOK, ANTI_TOK]
    _write_auth(agent, {
        "openrouter": {"type": "api_key", "key": OR_KEY},
        "ollama": {"type": "api_key", "key": OL_KEY},
        "google-vertex": {"type": "api_key", "key": VX_KEY},
        "anthropic": _oauth(CLAUDE_TOK),
        "openai-codex": _oauth(CODEX_TOK, accountId="ACC-1"),
        "antigravity": _oauth(ANTI_TOK),
    })
    with caplog.at_level(logging.DEBUG, logger="proxy"):
        proxy._sync_discovered_api_keys()
        proxy._resolve_openrouter_api_key()
        proxy._resolve_ollama_api_key()
        proxy._resolve_vertex_api_key()
        proxy.is_claude_auth_ready()
        proxy.claude_auth_source()
        proxy.ensure_codex_discovery()
        proxy.codex_auth_source()
        proxy.ensure_antigravity_discovery()
        proxy.antigravity_auth_source()
    blob = caplog.text
    for secret in secrets:
        assert secret not in blob

def test_no_network_when_not_ready(iso, monkeypatch):
    import aiohttp
    proxy, _ = iso

    def _boom(*a, **k):
        raise AssertionError("network attempted")

    monkeypatch.setattr(aiohttp, "ClientSession", _boom)
    assert proxy._resolve_openrouter_api_key() is None
    assert proxy.is_claude_auth_ready() is False
    assert proxy.ensure_codex_discovery() is False
    assert proxy.ensure_antigravity_discovery() is False

def test_health_reflects_discovery(test_client, iso):
    _proxy, agent = iso
    body = test_client.get("/health").json()
    assert body["openrouter_configured"] is False
    assert body["openrouter_source"] is None
    _write_auth(agent, {"openrouter": {"type": "api_key", "key": OR_KEY}})
    body = test_client.get("/health").json()
    assert body["openrouter_configured"] is True
    assert body["openrouter_source"] == "disk"
    assert OR_KEY not in json.dumps(body)
    _write_auth(agent, {"anthropic": _oauth(CLAUDE_TOK)})
    body = test_client.get("/health").json()
    assert body["claude"]["auth_cached"] is True
    assert body["claude"]["auth_source"] == "disk"

def test_registry_health_disk_ready(iso):
    proxy, agent = iso
    from proxy_internal import registry
    _write_auth(agent, {
        "openrouter": {"type": "api_key", "key": OR_KEY},
        "google-vertex": {"type": "api_key", "key": VX_KEY},
        "anthropic": _oauth(CLAUDE_TOK),
    })
    by_name = {p.name: p for p in registry.REGISTRY}
    assert by_name["openrouter"].health_fn() is True
    assert by_name["vertex"].health_fn() is True
    assert by_name["claude"].health_fn() is True
