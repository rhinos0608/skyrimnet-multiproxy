"""Wiring tests: disk-resolved OAuth client identity in the runtime.

No real tokens, no live provider calls. All identities are fixtures or
monkeypatched fakes; any unexpected network attempt raises.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

import proxy
from proxy_internal import cli_client_identity as ident
from proxy_internal.auth.antigravity import AntigravityAccount

FAKE_ID = "123456789012-abcdefghijklmnopqrstuvwxyz1234.apps.googleusercontent.com"
FAKE_SECRET = "GOCSPX-fakefakefakefakefake01"


@pytest.fixture()
def fresh_cache():
    """Reset proxy's lazy client slots; restore afterwards."""
    saved = (
        proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHE,
        proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHED,
        proxy._ANTIGRAVITY_OAUTH_SOURCE_LOGGED,
    )
    proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHE = None
    proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHED = False
    proxy._ANTIGRAVITY_OAUTH_SOURCE_LOGGED = False
    yield
    (
        proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHE,
        proxy._ANTIGRAVITY_OAUTH_CLIENT_CACHED,
        proxy._ANTIGRAVITY_OAUTH_SOURCE_LOGGED,
    ) = saved


@pytest.fixture()
def no_env(monkeypatch):
    for var in (
        "ANTIGRAVITY_CLIENT_ID",
        "ANTIGRAVITY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)


class _NoNetwork:
    def __init__(self, *args, **kwargs):
        raise AssertionError("network attempted while identity unresolved")


def test_import_does_not_scan_cli():
    script = (
        "import json;"
        "import proxy_internal.cli_client_identity as ident;"
        "calls={'scan':0,'antigravity':0,'from_disk':0};"
        "ident._scan_file=lambda *a,**k:(calls.__setitem__('scan',calls['scan']+1),None)[1];"
        "ident._from_disk=lambda *a,**k:(calls.__setitem__('from_disk',calls['from_disk']+1), (None,None))[1];"
        "orig_a=ident.resolve_antigravity_client;"
        "ident.resolve_antigravity_client=lambda *a,**k:(calls.__setitem__('antigravity',calls['antigravity']+1),None)[1];"
        "import proxy;"
        "print(json.dumps(calls))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    calls = json.loads(proc.stdout.strip().splitlines()[-1])
    assert calls == {"scan": 0, "antigravity": 0, "from_disk": 0}


def test_no_hardcoded_literal_attributes(fresh_cache, no_env, monkeypatch):
    monkeypatch.setattr(ident, "resolve_antigravity_client", lambda *a, **k: None)
    for name in (
        "ANTIGRAVITY_CLIENT_ID",
        "ANTIGRAVITY_CLIENT_SECRET",
    ):
        assert not hasattr(proxy, name), name


def test_accessor_caches_antigravity(fresh_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ident,
        "resolve_antigravity_client",
        lambda *a, **k: calls.append(1) or (FAKE_ID, FAKE_SECRET),
    )
    assert proxy.get_antigravity_oauth_client() == (FAKE_ID, FAKE_SECRET)
    assert proxy.get_antigravity_oauth_client() == (FAKE_ID, FAKE_SECRET)
    assert len(calls) == 1


def test_env_wins_without_disk_touch(fresh_cache, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_ID", FAKE_ID)
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_SECRET", FAKE_SECRET)

    def _boom(*args, **kwargs):
        raise AssertionError("disk scan must not run when env provides identity")

    monkeypatch.setattr(ident, "_from_disk", _boom)
    assert proxy.get_antigravity_oauth_client() == (FAKE_ID, FAKE_SECRET)


def test_unresolved_antigravity_fail_closed_no_network(fresh_cache, no_env, monkeypatch):
    monkeypatch.setattr(ident, "resolve_antigravity_client", lambda *a, **k: None)
    monkeypatch.setattr("aiohttp.ClientSession", _NoNetwork)
    acct = AntigravityAccount()
    acct.email = "someone@example.com"
    acct.refresh_token = "dummy-refresh-token"
    assert asyncio.run(acct._do_refresh()) is False
    assert acct.access_token is None
    err = proxy.antigravity_oauth_unresolved_error()
    assert "ANTIGRAVITY_CLIENT_ID" in err and "ANTIGRAVITY_CLIENT_SECRET" in err
    assert FAKE_SECRET not in err and "dummy-refresh-token" not in err
