"""Stale auth recapture: file first, matching discovery second, interactive last.

Isolated: tmp HOME + PI_CODING_AGENT_DIR, tmp auth-file paths, no real
machine auth, no live network (aiohttp guarded to raise).
"""
import json
import logging
from datetime import datetime, timedelta

import aiohttp
import pytest

FRESH_MS = 9999999999999
STALE_MS = 1000

ANTI_FILE_TOK = "anti-file-token-AAA"
ANTI_DISC_TOK = "anti-discovery-token-BBB"
SECRET_VALUES = (ANTI_FILE_TOK, ANTI_DISC_TOK)


@pytest.fixture()
def iso(monkeypatch, tmp_path, proxy_module):
    home = tmp_path / "home"
    agent = tmp_path / "agent"
    home.mkdir()
    agent.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    for env in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID"):
        monkeypatch.delenv(env, raising=False)
    # Point auth stores at tmp isolation (module globals were bound at import).
    monkeypatch.setattr(proxy_module, "ANTIGRAVITY_AUTH_FILE", str(tmp_path / "antigravity-auth.json"))
    monkeypatch.setattr(proxy_module, "_cfg", {})
    proxy_module._disk_source_logged.clear()
    # Fresh auth caches.
    from proxy_internal.auth.antigravity import AntigravityAccount
    from proxy_internal.auth.antigravity import AntigravityAuthCache
    monkeypatch.setattr(proxy_module, "antigravity_auth", AntigravityAuthCache())
    # Dummy session factory: creating sessions must not touch network.
    monkeypatch.setattr(proxy_module, "create_session", lambda *a, **k: object())
    # Zero-network guard: any real HTTP attempt fails loudly.
    class _NoNet:
        def __init__(self, *a, **k):
            raise AssertionError("network call attempted during recapture")
    monkeypatch.setattr(aiohttp, "ClientSession", _NoNet)
    return proxy_module, agent, tmp_path


def _write_auth(agent, blob):
    (agent / "auth.json").write_text(json.dumps(blob), encoding="utf-8")


def _oauth(access, expires=FRESH_MS):
    return {"type": "oauth", "access": access, "refresh": "REF", "expires": expires}


def _seed_stale_antigravity(proxy):
    from proxy_internal.auth.antigravity import AntigravityAccount
    stale = AntigravityAccount()
    stale.access_token = "stale-cached-anti"
    stale.refresh_token = None
    stale.email = "stale@example.com"
    stale.expires_at = datetime.now() - timedelta(hours=1)
    proxy.antigravity_auth.add_account(stale)


def _write_fresh_antigravity_file(proxy):
    payload = {"accounts": [{
        "refresh_token": "file-refresh-RT",
        "access_token": ANTI_FILE_TOK,
        "email": "file@example.com",
        "project_id": "p1",
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }]}
    with open(proxy.ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _assert_no_secrets(caplog):
    text = caplog.text
    for secret in SECRET_VALUES + ("stale-cached-anti", "file-refresh-RT"):
        assert secret not in text


@pytest.mark.asyncio
async def test_antigravity_stale_cache_fresh_file_no_network(iso, caplog):
    proxy, _agent, _tmp = iso
    from proxy_internal.auth.antigravity import recapture_antigravity_auth
    _seed_stale_antigravity(proxy)
    assert not proxy.antigravity_auth.is_ready
    _write_fresh_antigravity_file(proxy)
    with caplog.at_level(logging.INFO, logger="proxy"):
        source = await recapture_antigravity_auth()
    assert source == "file"  # zero network calls (guard would raise)
    assert proxy.antigravity_auth.is_ready
    assert "via file" in caplog.text
    _assert_no_secrets(caplog)


@pytest.mark.asyncio
async def test_antigravity_expired_discovery_stays_not_ready(iso, caplog):
    proxy, agent, _tmp = iso
    from proxy_internal.auth.antigravity import recapture_antigravity_auth
    _seed_stale_antigravity(proxy)
    _write_auth(agent, {"antigravity": _oauth(ANTI_DISC_TOK, expires=STALE_MS)})
    with caplog.at_level(logging.INFO, logger="proxy"):
        source = await recapture_antigravity_auth()
    assert source is None
    assert not proxy.antigravity_auth.is_ready
    _assert_no_secrets(caplog)


@pytest.mark.asyncio
async def test_cross_provider_isolation(iso, caplog):
    """A fresh record for another provider must not ready antigravity."""
    proxy, agent, _tmp = iso
    from proxy_internal.auth.antigravity import recapture_antigravity_auth
    _write_auth(agent, {"anthropic": _oauth("claude-other-provider-tok")})
    with caplog.at_level(logging.INFO, logger="proxy"):
        assert await recapture_antigravity_auth() is None
    assert not proxy.antigravity_auth.is_ready
    assert "claude-other-provider-tok" not in caplog.text


@pytest.mark.asyncio
async def test_no_interactive_demand_when_disk_usable(iso, caplog):
    proxy, agent, _tmp = iso
    from proxy_internal.auth.antigravity import load_antigravity_auth
    _write_fresh_antigravity_file(proxy)
    with caplog.at_level(logging.INFO, logger="proxy"):
        await load_antigravity_auth()
    assert proxy.antigravity_auth.is_ready
    assert "/config/antigravity-login" not in caplog.text
    _assert_no_secrets(caplog)


@pytest.mark.asyncio
async def test_late_login_picked_up_without_restart(iso, caplog):
    proxy, _agent, _tmp = iso
    from proxy_internal.auth.antigravity import recapture_antigravity_auth
    assert await recapture_antigravity_auth() is None
    _write_fresh_antigravity_file(proxy)
    with caplog.at_level(logging.INFO, logger="proxy"):
        assert await recapture_antigravity_auth() == "file"
    # Sync per-call ensure path is dynamic too.
    assert proxy.ensure_antigravity_discovery() is True
    assert proxy.antigravity_auth_source() in ("config", "file")
    _assert_no_secrets(caplog)


@pytest.mark.asyncio
async def test_discovery_source_logged_redacted(iso, caplog):
    proxy, agent, _tmp = iso
    from proxy_internal.auth.antigravity import recapture_antigravity_auth
    _write_auth(agent, {
        "antigravity": _oauth(ANTI_DISC_TOK),
    })
    with caplog.at_level(logging.INFO, logger="proxy"):
        assert await recapture_antigravity_auth() == "discovery"
    assert "via discovery" in caplog.text
    _assert_no_secrets(caplog)
