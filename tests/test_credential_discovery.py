"""Tests for proxy_internal.credential_discovery (no real files touched)."""

import json
import logging
import os
import stat

import pytest

from proxy_internal import credential_discovery as cd

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
    monkeypatch.delenv("TEST_DISC_KEY", raising=False)
    return home, agent

def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content), encoding="utf-8")
    return path

def _snapshot(path):
    return path.read_bytes() if path.exists() else None

def test_command_code_pi_api_key_first(isolated):
    home, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "PI-KEY"}})
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    _write(agent / "settings.json", {"command-code.apiKey": "SETTINGS-KEY"})
    assert cd.discover_command_code_key() == "PI-KEY"

def test_command_code_settings_before_cli(isolated):
    home, agent = isolated
    _write(agent / "settings.json", {"command-code.apiKey": "SETTINGS-KEY"})
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    assert cd.discover_command_code_key() == "SETTINGS-KEY"

def test_command_code_nested_settings(isolated):
    _, agent = isolated
    _write(agent / "settings.json", {"command-code": {"apiKey": "NESTED-KEY"}})
    assert cd.discover_command_code_key() == "NESTED-KEY"

def test_command_code_cli_fallback(isolated):
    home, _ = isolated
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY"})
    assert cd.discover_command_code_key() == "CLI-KEY"

def test_command_code_synthetic_oauth_accepted(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        {
            "command-code": {
                "type": "oauth",
                "access": "SYNTH-KEY",
                "refresh": "SYNTH-KEY",
                "expires": SYNTH,
            }
        },
    )
    assert cd.discover_command_code_key() == "SYNTH-KEY"

def test_command_code_real_oauth_rejected(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        {
            "command-code": {
                "type": "oauth",
                "access": "REAL-ACCESS",
                "refresh": "REAL-REFRESH",
                "expires": 9999999999,
            }
        },
    )
    assert cd.discover_command_code_key() is None

def test_command_code_oauth_mismatch_rejected(isolated):
    _, agent = isolated
    rec = {"type": "oauth", "access": "AAA", "refresh": "BBB", "expires": SYNTH}
    _write(agent / "auth.json", {"command-code": rec})
    assert cd.discover_command_code_key() is None

def test_provider_isolation_command_code_ignores_opencode(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        {
            "opencode": {"type": "api_key", "key": "WRONG-PROVIDER"},
            "opencode-go": {"type": "api_key", "key": "WRONG-PROVIDER-GO"},
        },
    )
    assert cd.discover_command_code_key() is None

def test_provider_isolation_opencode_ignores_command_code(isolated):
    home, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "WRONG"}})
    _write(home / ".commandcode" / "auth.json", {"apiKey": "WRONG"})
    assert cd.discover_opencode_key("Go") is None
    assert cd.discover_opencode_key("Zen") is None

def test_opencode_plan_preference(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        {
            "opencode": {"type": "api_key", "key": "ZEN-KEY"},
            "opencode-go": {"type": "api_key", "key": "GO-KEY"},
        },
    )
    assert cd.discover_opencode_key("Go") == "GO-KEY"
    assert cd.discover_opencode_key("Zen") == "ZEN-KEY"

def test_opencode_shared_key_fallback(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"opencode": {"type": "api_key", "key": "SHARED"}})
    assert cd.discover_opencode_key("Go") == "SHARED"
    assert cd.discover_opencode_key("Zen") == "SHARED"

def test_opencode_cli_plan_preference(isolated):
    home, _ = isolated
    _write(
        home / ".local" / "share" / "opencode" / "auth.json",
        {
            "opencode": {"type": "api", "key": "CLI-ZEN"},
            "opencode-go": {"type": "api", "key": "CLI-GO"},
        },
    )
    assert cd.discover_opencode_key("Go") == "CLI-GO"
    assert cd.discover_opencode_key("Zen") == "CLI-ZEN"

def test_opencode_cli_wrong_type_rejected(isolated):
    home, _ = isolated
    _write(
        home / ".local" / "share" / "opencode" / "auth.json",
        {"opencode-go": {"type": "oauth", "key": "NOPE"}},
    )
    assert cd.discover_opencode_key("Go") is None

def test_pi_beats_cli_for_opencode(isolated):
    home, agent = isolated
    _write(agent / "auth.json", {"opencode-go": {"type": "api_key", "key": "PI-GO"}})
    _write(
        home / ".local" / "share" / "opencode" / "auth.json",
        {"opencode-go": {"type": "api", "key": "CLI-GO"}},
    )
    assert cd.discover_opencode_key("Go") == "PI-GO"

def test_command_valued_key_never_used(isolated):
    _, agent = isolated
    blob = {"command-code": {"type": "api_key", "key": "!echo pwned"}}
    _write(agent / "auth.json", blob)
    assert cd.discover_command_code_key() is None

def test_non_string_key_rejected(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": 12345}})
    assert cd.discover_command_code_key() is None

def test_malformed_files_ignored(isolated):
    home, agent = isolated
    _write(agent / "auth.json", "not json at all")
    _write(agent / "settings.json", "[1,2,3]")
    _write(home / ".commandcode" / "auth.json", "definitely not json")
    assert cd.discover_command_code_key() is None
    assert cd.discover_opencode_key("Go") is None

def test_missing_files_return_none(isolated):
    assert cd.discover_command_code_key() is None
    assert cd.discover_opencode_key("Zen") is None

def test_world_writable_ignored_posix(isolated):
    if os.name != "posix":
        pytest.skip("POSIX-only")
    _, agent = isolated
    p = _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "K"}})
    os.chmod(p, 0o666 | stat.S_IWOTH)
    assert cd.discover_command_code_key() is None

def test_exact_env_reference_resolved(isolated, monkeypatch):
    _, agent = isolated
    monkeypatch.setenv("TEST_DISC_KEY", "ENV-KEY")
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "$TEST_DISC_KEY"}})
    assert cd.discover_command_code_key() == "ENV-KEY"

def test_env_braced_reference_resolved(isolated, monkeypatch):
    _, agent = isolated
    monkeypatch.setenv("TEST_DISC_KEY", "ENV-KEY")
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "${TEST_DISC_KEY}"}})
    assert cd.discover_command_code_key() == "ENV-KEY"

def test_unresolved_env_ignored(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "$TEST_DISC_KEY"}})
    assert cd.discover_command_code_key() is None

def test_embedded_template_ignored(isolated, monkeypatch):
    _, agent = isolated
    monkeypatch.setenv("TEST_DISC_KEY", "ENV-KEY")
    blob = {"command-code": {"type": "api_key", "key": "prefix-$TEST_DISC_KEY"}}
    _write(agent / "auth.json", blob)
    assert cd.discover_command_code_key() is None

def test_no_values_or_paths_logged(isolated, caplog):
    _, agent = isolated
    _write(agent / "auth.json", "still not json")
    with caplog.at_level(logging.DEBUG, logger="proxy"):
        cd.discover_command_code_key()
        cd.discover_opencode_key("Go")
    for record in caplog.records:
        assert str(agent) not in record.getMessage()

def test_read_only_no_writes(isolated):
    home, agent = isolated
    auth = _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "K1"}})
    cli = _write(home / ".commandcode" / "auth.json", {"apiKey": "K2"})
    before_auth = _snapshot(auth)
    before_cli = _snapshot(cli)
    cd.discover_command_code_key()
    cd.discover_opencode_key("Go")
    assert _snapshot(auth) == before_auth
    assert _snapshot(cli) == before_cli

def test_whitespace_only_key_rejected(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "   "}})
    assert cd.discover_command_code_key() is None

def test_padded_command_value_never_executed(isolated):
    _, agent = isolated
    blob = {"command-code": {"type": "api_key", "key": "  !echo pwned"}}
    _write(agent / "auth.json", blob)
    assert cd.discover_command_code_key() is None

def test_control_chars_rejected_pi_auth(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "KEY\nINJECTED"}})
    assert cd.discover_command_code_key() is None
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "KEY\r\nX"}})
    assert cd.discover_command_code_key() is None
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "KEY\x00X"}})
    assert cd.discover_command_code_key() is None

def test_control_chars_rejected_cli_stores(isolated):
    home, agent = isolated
    _write(home / ".commandcode" / "auth.json", {"apiKey": "CLI-KEY\n"})
    assert cd.discover_command_code_key() is None
    _write(
        home / ".local" / "share" / "opencode" / "auth.json",
        {"opencode-go": {"type": "api", "key": "GO-KEY\r\n"}},
    )
    assert cd.discover_opencode_key("Go") is None

def test_opencode_control_chars_rejected_pi(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"opencode-go": {"type": "api_key", "key": "GO\tKEY"}})
    assert cd.discover_opencode_key("Go") is None

def test_padded_env_expression_after_trim_rejected(isolated, monkeypatch):
    _, agent = isolated
    monkeypatch.setenv("TEST_DISC_KEY", "ENV-KEY")
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "  prefix-$TEST_DISC_KEY  "}})
    assert cd.discover_command_code_key() is None

def test_padded_exact_env_reference_resolved(isolated, monkeypatch):
    _, agent = isolated
    monkeypatch.setenv("TEST_DISC_KEY", "ENV-KEY")
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "  $TEST_DISC_KEY  "}})
    assert cd.discover_command_code_key() == "ENV-KEY"

def test_padded_literal_key_trimmed(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"command-code": {"type": "api_key", "key": "  PI-KEY  "}})
    assert cd.discover_command_code_key() == "PI-KEY"

def test_synthetic_oauth_malformed_rejected(isolated):
    _, agent = isolated
    for bad in ("   ", "KEY\nX", "  !echo pwned", "KEY\rX", "KEY\x7fX"):
        _write(
            agent / "auth.json",
            {"command-code": {"type": "oauth", "access": bad, "refresh": bad, "expires": SYNTH}},
        )
        assert cd.discover_command_code_key() is None

def test_direct_settings_invalid_falls_back_to_nested(isolated):
    _, agent = isolated
    _write(
        agent / "settings.json",
        {"command-code.apiKey": "   ", "command-code": {"apiKey": "NESTED-KEY"}},
    )
    assert cd.discover_command_code_key() == "NESTED-KEY"

# ---- new provider discovery ----

FRESH = 9999999999999
STALE = 1000

def _oauth(provider, access="TOK", refresh="REF", expires=FRESH, **extra):
    rec = {"type": "oauth", "access": access, "refresh": refresh, "expires": expires}
    rec.update(extra)
    return {provider: rec}

def test_vertex_happy_and_isolation(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"google-vertex": {"type": "api_key", "key": "VTX"}})
    assert cd.discover_vertex_key() == "VTX"
    _write(agent / "auth.json", {"openrouter": {"type": "api_key", "key": "OR"}})
    assert cd.discover_vertex_key() is None

def test_openrouter_happy_and_rejects_oauth(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"openrouter": {"type": "api_key", "key": "OR-KEY"}})
    assert cd.discover_openrouter_key() == "OR-KEY"
    _write(agent / "auth.json", _oauth("openrouter"))
    assert cd.discover_openrouter_key() is None

def test_ollama_happy_and_isolation(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"ollama": {"type": "api_key", "key": "OLL"}})
    assert cd.discover_ollama_key() == "OLL"
    _write(agent / "auth.json", {"openrouter": {"type": "api_key", "key": "OR"}})
    assert cd.discover_ollama_key() is None

def _oauth_happy_and_expired(isolated_fn, provider, fn):
    _, agent = isolated_fn
    _write(agent / "auth.json", _oauth(provider))
    got = fn()
    token = got["access_token"] if isinstance(got, dict) else got
    assert token == "TOK"
    _write(agent / "auth.json", _oauth(provider, expires=STALE))
    assert fn() is None

def test_claude_oauth(isolated):
    home, agent = isolated
    _oauth_happy_and_expired((home, agent), "anthropic", cd.discover_claude_oauth)

def test_codex_oauth_dict(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        _oauth("openai-codex", accountId="ACC-1"),
    )
    got = cd.discover_codex_oauth()
    assert got == {"access_token": "TOK", "account_id": "ACC-1"}
    _write(agent / "auth.json", _oauth("openai-codex", expires=STALE))
    assert cd.discover_codex_oauth() is None

def test_antigravity_oauth(isolated):
    home, agent = isolated
    _oauth_happy_and_expired((home, agent), "antigravity", cd.discover_antigravity_oauth)

def test_oauth_wrong_type_rejected(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"anthropic": {"type": "api_key", "key": "K"}})
    assert cd.discover_claude_oauth() is None
    _write(agent / "auth.json", {"openai-codex": {"type": "api_key", "key": "K"}})
    assert cd.discover_codex_oauth() is None

def test_oauth_malformed_expires_rejected(isolated):
    _, agent = isolated
    for bad in (None, "soon", True, [1]):
        _write(
            agent / "auth.json",
            {"anthropic": {"type": "oauth", "access": "T", "refresh": "R", "expires": bad}},
        )
        assert cd.discover_claude_oauth() is None

def test_new_fns_bang_and_control_rejected(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"openrouter": {"type": "api_key", "key": "A\nB"}})
    assert cd.discover_openrouter_key() is None
    _write(agent / "auth.json", _oauth("anthropic", access="!cmd", refresh="!cmd"))
    assert cd.discover_claude_oauth() is None
    _write(agent / "auth.json", _oauth("anthropic", access="A\x00B", refresh="R"))
    assert cd.discover_claude_oauth() is None

def test_new_fns_malformed_json(isolated):
    _, agent = isolated
    _write(agent / "auth.json", "not json here either")
    assert cd.discover_vertex_key() is None
    assert cd.discover_openrouter_key() is None
    assert cd.discover_ollama_key() is None
    assert cd.discover_claude_oauth() is None
    assert cd.discover_codex_oauth() is None
    assert cd.discover_antigravity_oauth() is None

def test_new_fns_world_writable_ignored(isolated):
    if os.name != "posix":
        pytest.skip("POSIX-only")
    _, agent = isolated
    p = _write(agent / "auth.json", {"google-vertex": {"type": "api_key", "key": "K"}})
    os.chmod(p, 0o666 | stat.S_IWOTH)
    assert cd.discover_vertex_key() is None
    assert cd.discover_claude_oauth() is None

def test_cross_provider_isolation_new(isolated):
    _, agent = isolated
    _write(agent / "auth.json", {"google-vertex": {"type": "api_key", "key": "VTX"}})
    assert cd.discover_openrouter_key() is None
    assert cd.discover_ollama_key() is None
    _write(agent / "auth.json", _oauth("anthropic"))
    assert cd.discover_antigravity_oauth() is None
    assert cd.discover_codex_oauth() is None

def test_discover_all_statuses_no_secrets(isolated):
    _, agent = isolated
    _write(
        agent / "auth.json",
        {
            "google-vertex": {"type": "api_key", "key": "VTX-SECRET"},
            "anthropic": {"type": "oauth", "access": "TOK", "refresh": "R", "expires": FRESH},
        },
    )
    status = cd.discover_all()
    assert status["vertex"] == "ok"
    assert status["claude"] == "ok"
    assert status["codex"] == "missing"
    blob = json.dumps(status)
    assert "VTX-SECRET" not in blob
    assert "TOK" not in blob or blob.count("TOK") == 0
    assert set(status.values()) <= {"ok", "expired", "missing"}

def test_discover_all_no_writes(isolated):
    home, agent = isolated
    auth = _write(agent / "auth.json", {"google-vertex": {"type": "api_key", "key": "K"}})
    cli = _write(
        home / ".local" / "share" / "opencode" / "auth.json",
        {"opencode-go": {"type": "api", "key": "C"}},
    )
    before = (_snapshot(auth), _snapshot(cli))
    cd.discover_all()
    cd.discover_vertex_key()
    cd.discover_openrouter_key()
    cd.discover_ollama_key()
    cd.discover_claude_oauth()
    cd.discover_codex_oauth()
    cd.discover_antigravity_oauth()
    assert (_snapshot(auth), _snapshot(cli)) == before
