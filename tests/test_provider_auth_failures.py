"""Credential-side provider failure guards (fakes only, no network/credentials).

Covers three live failures:
- opencode-go 403: a stale cross-plan global must not shadow the
  plan-matched discovered key.
- command_api 400: only exact 403/upgrade_required may trigger CLI
  fallback; other upstreams (e.g. 400 insufficient-credits) must not.
- antigravity 401: headers match the official client's wire fingerprint;
  Pi JSON-blob OAuth values are unwrapped, never sent wholesale.
"""
import json

import proxy
from proxy_internal.auth import antigravity as ag
from proxy_internal.providers import command_code as cc
from proxy_internal.providers import opencode as oc

# --- opencode plan-matched selection ---

def test_go_prefers_discovered_go_key_over_cli_shadowed_global(proxy_module, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(proxy_module, "opencode_go_api_key", "STALE-SHARED-KEY", raising=False)
    monkeypatch.setattr(proxy_module, "opencode_api_key", "ZEN-KEY", raising=False)
    monkeypatch.setattr(disc, "discover_opencode_key",
                        lambda plan: "GO-PLAN-KEY" if plan == "Go" else "ZEN-KEY")
    # Global merely mirrors the CLI auth file (startup auto-load) -> yields.
    monkeypatch.setattr(oc, "_cli_file_key",
                        lambda entry: "STALE-SHARED-KEY" if entry == "opencode-go" else "ZEN-KEY")
    _, url, key, plan = oc._resolve_opencode("opencode-go:glm-5.3")
    assert plan == "Go"
    assert url == proxy_module.OPENCODE_GO_URL
    assert key == "GO-PLAN-KEY"


def test_explicit_global_beats_discovery(proxy_module, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(proxy_module, "opencode_go_api_key", "OPERATOR-GO", raising=False)
    monkeypatch.setattr(proxy_module, "opencode_api_key", None, raising=False)
    monkeypatch.setattr(disc, "discover_opencode_key", lambda plan: "DISC-KEY")
    # Operator value is not what the CLI file holds -> explicit, wins.
    monkeypatch.setattr(oc, "_cli_file_key", lambda entry: "CLI-FILE-KEY")
    _, _, key, _ = oc._resolve_opencode("opencode-go:glm-5.3")
    assert key == "OPERATOR-GO"

def test_zen_prefers_discovered_zen_key_over_stale_global(proxy_module, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(proxy_module, "opencode_api_key", "STALE-SHARED-KEY", raising=False)
    monkeypatch.setattr(proxy_module, "opencode_go_api_key", "GO-KEY", raising=False)
    monkeypatch.setattr(disc, "discover_opencode_key",
                        lambda plan: "GO-KEY" if plan == "Go" else "ZEN-PLAN-KEY")
    monkeypatch.setattr(oc, "_cli_file_key",
                        lambda entry: "STALE-SHARED-KEY" if entry == "opencode" else "GO-KEY")
    _, url, key, plan = oc._resolve_opencode("opencode:glm-5.3")
    assert plan == "Zen"
    assert url == proxy_module.OPENCODE_ZEN_URL
    assert key == "ZEN-PLAN-KEY"

def test_explicit_cfg_key_wins_over_discovery(proxy_module, monkeypatch):
    import proxy_internal.credential_discovery as disc
    monkeypatch.setattr(proxy_module, "opencode_go_api_key", None, raising=False)
    monkeypatch.setattr(proxy_module, "opencode_api_key", None, raising=False)
    monkeypatch.setattr(disc, "discover_opencode_key", lambda plan: "DISC-KEY")
    monkeypatch.setitem(proxy_module._cfg, "opencode_go_api_key", "EXPLICIT-GO-KEY")
    try:
        _, _, key, _ = oc._resolve_opencode("opencode-go:glm-5.3")
        assert key == "EXPLICIT-GO-KEY"
    finally:
        monkeypatch.setitem(proxy_module._cfg, "opencode_go_api_key", None)

def test_go_headers_carry_bearer_and_session():
    headers = oc._build_opencode_headers("K", "Go", session_id="s1")
    assert headers["Authorization"] == "Bearer K"
    assert headers["x-opencode-session"] == "s1"
    assert "skyrimnet" in headers["User-Agent"]

# --- command_api fallback trigger stays narrow ---

def test_insufficient_credits_is_not_upgrade_required():
    body = json.dumps({"error": {"message": "You have insufficient credits.",
                                  "type": "invalid_request_error",
                                  "code": "BAD_REQUEST"}})
    assert cc._is_upgrade_required(body) is False

def test_exact_upgrade_required_still_detected():
    assert cc._is_upgrade_required(json.dumps({"code": "upgrade_required"})) is True
    assert cc._is_upgrade_required(
        json.dumps({"error": {"code": "upgrade_required"}})) is True
    assert cc._is_upgrade_required("not json") is False

# --- antigravity wire fingerprint + token unwrap ---

def test_antigravity_headers_match_official_client():
    headers = ag.get_antigravity_headers()
    assert headers["User-Agent"] == (
        "antigravity/cli/1.2.12 (aidev_client; os_type=linux; arch=amd64; "
        "cl=989022288; auth_method=consumer)")
    assert "Client-Metadata" not in headers
    assert "X-Goog-Api-Client" not in headers

def test_auth_headers_carry_bearer_only_plus_fingerprint(proxy_module):
    account = ag.AntigravityAccount()
    account.access_token = "ya29.fake"
    headers = account.get_auth_headers()
    assert headers["Authorization"] == "Bearer ya29.fake"
    assert "Client-Metadata" not in headers
    assert "X-Goog-Api-Client" not in headers

def test_adopt_unwraps_json_blob_with_project(proxy_module):
    blob = json.dumps({"token": "ya29.nested-fake", "projectId": "aicode-test"})
    assert ag.adopt_discovered_antigravity_token(blob, proxy_module=proxy_module) is True
    matches = [a for a in proxy_module.antigravity_auth.accounts
                if a.email == ag.ANTIGRAVITY_DISCOVERY_EMAIL]
    assert len(matches) == 1
    assert matches[0].access_token == "ya29.nested-fake"
    assert matches[0].project_id == "aicode-test"

def test_adopt_rejects_garbage_token(proxy_module):
    assert ag.adopt_discovered_antigravity_token("", proxy_module=proxy_module) is False
    assert ag.adopt_discovered_antigravity_token(
        '{"token": "", "projectId": "x"}', proxy_module=proxy_module) is False
    assert ag.adopt_discovered_antigravity_token(
        '{"token": 123}', proxy_module=proxy_module) is False
    assert ag.adopt_discovered_antigravity_token(
        '{"nope": 1}', proxy_module=proxy_module) is False

def test_fallback_project_is_aicode_consumers():
    assert proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID == "aicode-consumers"
    assert proxy.ANTIGRAVITY_ENDPOINT == "https://daily-cloudcode-pa.googleapis.com"
