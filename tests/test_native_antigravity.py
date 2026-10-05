"""Native agy session reuse: synthetic Keychain records only, no live credentials."""
import asyncio
import base64
import json
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import proxy
from proxy_internal import credential_discovery as discovery
from proxy_internal.auth import antigravity as auth

def run(coro):
    # Do not clear the policy loop used by older synchronous tests.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def record(access="fixture-access", expiry=None):
    return {"token": {"access_token": access, "refresh_token": "fixture-refresh",
                        "token_type": "Bearer", "expiry": expiry or
                        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            "id_token": "fixture-id-token"}

@pytest.fixture
def native(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "native_antigravity_enabled", lambda: True)
    monkeypatch.setattr(auth, "native_antigravity_enabled", lambda: True)
    responses = [record()]
    calls = []

    def keychain(args, **kwargs):
        calls.append((args, kwargs))
        raw = json.dumps(responses[-1]).encode()
        return SimpleNamespace(returncode=0, stdout=b"go-keyring-base64:" + base64.b64encode(raw))

    monkeypatch.setattr(discovery.subprocess, "run", keychain)
    monkeypatch.setattr(proxy, "antigravity_auth", auth.AntigravityAuthCache())
    monkeypatch.setattr(proxy, "ANTIGRAVITY_AUTH_FILE", str(tmp_path / "auth.json"))
    return responses, calls

def test_native_is_default_on_macos(monkeypatch):
    monkeypatch.setattr(discovery.sys, "platform", "darwin")
    assert discovery.native_antigravity_enabled()

def test_keychain_read_is_scoped_and_discards_refresh_token(native):
    token = discovery.discover_native_antigravity_token()
    assert token.access_token == "fixture-access"
    assert token.expires_at.tzinfo is not None
    assert not hasattr(token, "refresh_token")
    assert "fixture-access" not in repr(token)
    _, calls = native
    args, kwargs = calls[0]
    assert args == ["/usr/bin/security", "find-generic-password", "-s", "gemini", "-a", "antigravity", "-w"]
    assert kwargs["timeout"] <= 10
    assert kwargs.get("shell", False) is False
    assert kwargs["stdin"] is subprocess.DEVNULL

@pytest.mark.parametrize("bad", [None, [], {}, {"token": []},
    record(access=""), record(access="unsafe\nheader"), record(expiry="invalid"),
    record(expiry="2020-01-01T00:00:00+00:00"), record(expiry="2099-01-01T00:00:00")])
def test_invalid_or_expired_native_record_rejected(native, bad):
    native[0].append(bad)
    assert discovery.discover_native_antigravity_token() is None

@pytest.mark.parametrize("raw", [b"go-keyring-base64:!!", b"not-json", b"x" * (256*1024+1)])
def test_bad_encoding_or_oversize_rejected(native, monkeypatch, raw):
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=raw))
    assert discovery.discover_native_antigravity_token() is None

def test_lookup_timeout_redacts_exception(native, monkeypatch, caplog):
    def fail(*a, **k):
        raise subprocess.TimeoutExpired("fixture-secret", 8, output=b"fixture-secret")
    monkeypatch.setattr(discovery.subprocess, "run", fail)
    assert discovery.discover_native_antigravity_token() is None
    assert "fixture-secret" not in caplog.text

def test_startup_ignores_existing_proxy_store(native, monkeypatch, tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("existing stored OAuth credentials")
    def forbidden(*a, **k):
        raise AssertionError("old OAuth/discovery path must not run")
    monkeypatch.setattr(auth, "_load_antigravity_file", forbidden)
    monkeypatch.setattr(auth, "_adopt_discovered_antigravity_token", forbidden)
    monkeypatch.setattr(proxy, "get_antigravity_oauth_client", forbidden)
    run(auth.load_antigravity_auth())
    account = proxy.antigravity_auth.get_next_account()
    assert account.access_token == "fixture-access"
    assert account.refresh_token is None
    assert proxy.antigravity_auth.is_ready
    assert proxy.antigravity_auth_source() == "keychain"
    auth._save_antigravity_auth()
    assert path.read_text() == "existing stored OAuth credentials"
    assert auth._snapshot_antigravity_auth_accounts() == []

def test_native_rotation_and_logout_never_refresh_or_persist(native, monkeypatch):
    run(auth.load_antigravity_auth())
    account = proxy.antigravity_auth.get_next_account()
    native[0].append(record("rotated-access"))
    def forbidden(*a, **k):
        raise AssertionError("must not mint or persist native tokens")
    monkeypatch.setattr(proxy, "get_antigravity_oauth_client", forbidden)
    monkeypatch.setattr(auth, "_write_antigravity_auth_snapshot", forbidden)
    assert run(proxy.antigravity_auth.refresh_if_needed())
    assert account.access_token == "rotated-access"
    native[0].append({})
    assert not run(proxy.antigravity_auth.refresh_if_needed())
    assert account.access_token is None
    assert not proxy.antigravity_auth.is_ready
    assert not proxy.ensure_antigravity_discovery()
    assert run(auth.recapture_antigravity_auth()) is None

def test_login_endpoint_reuses_native_without_browser(native, monkeypatch):
    from proxy_internal.endpoints.antigravity_oauth import antigravity_login
    def forbidden(*a, **k):
        raise AssertionError("must not open callback server")
    monkeypatch.setattr(proxy, "start_oauth_callback_server", forbidden)
    response = run(antigravity_login())
    assert response.status_code == 200
    assert b"agy" in response.body
    assert b"accounts.google.com" not in response.body
    native[0].append({})
    assert run(antigravity_login()).status_code == 503

@pytest.mark.parametrize("encoding", ["plain", "hex"])
def test_keyring_legacy_encodings(native, monkeypatch, encoding):
    raw = json.dumps(record()).encode()
    if encoding == "hex":
        raw = b"go-keyring-encoded:" + raw.hex().encode()
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=raw))
    assert discovery.discover_native_antigravity_token().access_token == "fixture-access"


def test_non_macos_never_reads_keychain(monkeypatch):
    monkeypatch.setattr(discovery.sys, "platform", "linux")
    def forbidden(*a, **k):
        raise AssertionError("Keychain is macOS only")
    monkeypatch.setattr(discovery.subprocess, "run", forbidden)
    assert discovery.discover_native_antigravity_token() is None


def test_provider_401_rereads_native_token(native):
    from proxy_internal.providers.antigravity_structured import call_antigravity_structured
    run(auth.load_antigravity_auth())
    sent = []

    class Response:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            return "unauthorized"

        async def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

    def post(url, *, json, headers):
        sent.append(headers["Authorization"])
        if len(sent) == 1:
            native[0].append(record("rotated-access"))
            return Response(401)
        return Response(200)

    proxy.antigravity_auth.accounts[0].session = SimpleNamespace(post=post)
    result = run(call_antigravity_structured(
        {"api_model": "gemini-3-flash", "endpoint_url": "https://fixture.invalid"},
        {"messages": [{"role": "user", "content": "Reply OK"}]}, "native-test",
    ))
    assert sent == ["Bearer fixture-access", "Bearer rotated-access"]
    assert result["choices"][0]["message"]["content"] == "OK"


@pytest.mark.parametrize("entry", ["messages", "chat"])
def test_request_entry_recaptures_expired_native_session(native, monkeypatch, entry):
    run(auth.load_antigravity_auth())
    account = proxy.antigravity_auth.accounts[0]
    account.expires_at = datetime.now() - timedelta(seconds=1)
    account.session = SimpleNamespace()
    native[0].append(record("rotated-access"))
    assert not proxy.antigravity_auth.is_ready
    if entry == "messages":
        resolved = run(proxy._resolve_oai_compatible("antigravity-gemini-3-flash"))
        assert resolved["headers"]["Authorization"] == "Bearer rotated-access"
    else:
        async def reply(*args, **kwargs):
            assert account.access_token == "rotated-access"
            return "OK"
        monkeypatch.setattr(proxy, "call_antigravity_direct", reply)
        result = run(proxy._chat_completions_inner(proxy.ChatRequest(
            model="antigravity-gemini-3-flash", stream=False,
            messages=[{"role": "user", "content": "Reply OK"}],
        )))
        assert result["choices"][0]["message"]["content"] == "OK"


def test_native_account_cannot_be_removed_by_dashboard(native):
    from proxy_internal.endpoints.antigravity_oauth import remove_antigravity_account
    run(auth.load_antigravity_auth())
    response = run(remove_antigravity_account("agy-keychain"))
    assert response.status_code == 409
    assert proxy.antigravity_auth.is_ready
