"""Tests for proxy_internal.cli_client_identity (fixtures only, no real CLI)."""

from __future__ import annotations

import logging
import os

import pytest

from proxy_internal import cli_client_identity as ident

FAKE_ID = "123456789012-abcdefghijklmnop123456.apps.googleusercontent.com"
FAKE_SECRET = "GOCSPX-fakeSecretChars123456"
DECOY_ID = "123456789012-library-sample-value.apps.googleusercontent.com"
SECOND_ID = "987654321098-zyxwvutsrqponmlkjihg.z123.apps.googleusercontent.com"
SECOND_SECRET = "GOCSPX-secondFakeSecretValue9"


def _write(path, data: bytes) -> str:
    p = str(path)
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_disk_extraction_pair(tmp_path):
    p = _write(tmp_path / "agy-fake", b"\x00bin\xff" + FAKE_ID.encode() + b"\x00pad" + FAKE_SECRET.encode() + b"\x00")
    assert ident.resolve_antigravity_client([p], allow_env=False) == (FAKE_ID, FAKE_SECRET)
    assert ident.resolve("antigravity", [p], allow_env=False) == (FAKE_ID, FAKE_SECRET)
    assert ident.client_source("antigravity", [p], allow_env=False) == "cli-path"


def test_decoy_only_yields_none(tmp_path):
    p = _write(tmp_path / "gemini-fake", b"lib " + DECOY_ID.encode() + b" google-auth-library docs")
    assert ident.resolve("gemini", [p], allow_env=False) is None
    assert ident.client_source("gemini", [p], allow_env=False) == "none"


def test_id_without_secret_none(tmp_path):
    p = _write(tmp_path / "noid", b"xx" + FAKE_ID.encode() + b" trailing no secret here")
    assert ident.resolve_antigravity_client([p], allow_env=False) is None


def test_secret_without_id_none(tmp_path):
    p = _write(tmp_path / "nosec", b"xx " + FAKE_SECRET.encode() + b" no id here")
    assert ident.resolve_antigravity_client([p], allow_env=False) is None


def test_env_override_wins_over_disk(tmp_path, monkeypatch):
    p = _write(tmp_path / "disk", b"xx" + FAKE_ID.encode() + b" " + FAKE_SECRET.encode())
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_ID", "env-id-1.apps.googleusercontent.com")
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_SECRET", "GOCSPX-envsecret12345678")
    pair = ident.resolve_antigravity_client([p])
    assert pair == ("env-id-1.apps.googleusercontent.com", "GOCSPX-envsecret12345678")
    assert ident.client_source("antigravity", [p]) == "env"


def test_missing_cli_path_none(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    assert ident.resolve_antigravity_client([missing], allow_env=False) is None
    assert ident.resolve("gemini", [missing], allow_env=False) is None
    assert ident.client_source("antigravity", [missing], allow_env=False) == "none"


def test_unreadable_file_none(tmp_path):
    p = _write(tmp_path / "locked", b"xx" + FAKE_ID.encode() + b" " + FAKE_SECRET.encode())
    os.chmod(p, 0x000)
    try:
        if os.access(p, os.R_OK):
            pytest.skip("running as root; unreadable bit not enforced")
        assert ident.resolve_antigravity_client([p], allow_env=False) is None
    finally:
        os.chmod(p, 0o600)


def test_garbage_oversized_bounded(tmp_path):
    p = _write(tmp_path / "garbage", os.urandom(2 * 1024 * 1024))
    assert ident.resolve_antigravity_client([p], allow_env=False) is None


def test_no_credential_logged(tmp_path, caplog):
    p = _write(tmp_path / "logged", b"xx" + FAKE_ID.encode() + b" " + FAKE_SECRET.encode())
    with caplog.at_level(logging.DEBUG, logger="proxy"):
        pair = ident.resolve_antigravity_client([p], allow_env=False)
        ident.client_source("antigravity", [p], allow_env=False)
    assert pair == (FAKE_ID, FAKE_SECRET)
    for record in caplog.records:
        text = record.getMessage()
        assert FAKE_ID not in text
        assert FAKE_SECRET not in text


def test_unknown_provider_none():
    assert ident.resolve("nope", [], allow_env=False) is None
    assert ident.client_source("nope", [], allow_env=False) == "none"


@pytest.mark.skip(reason="opt-in: hits real installed CLI; run manually")
def test_real_antigravity_cli_opt_in():
    pair = ident.resolve_antigravity_client()
    assert pair is None or len(pair) == 2


def test_several_clients_need_a_verifier(tmp_path):
    """A file with two ids and two secrets must not be silently mis-paired.

    Mirrors the layout of the installed ``agy`` binary: both ids first, then
    both secrets back-to-back, so file order says nothing about which id owns
    which secret.
    """
    p = _write(
        tmp_path / "agy-fake",
        FAKE_ID.encode() + b"\x00" + SECOND_ID.encode() + b"\x00"
        + SECOND_SECRET.encode() + FAKE_SECRET.encode() + b"\x00",
    )
    # Unverified: first candidate (a guess, kept for backward compatibility).
    assert ident.resolve_antigravity_client([p], allow_env=False) == (FAKE_ID, SECOND_SECRET)

    probed = []

    def verifier(pair):
        probed.append(pair)
        return pair == (SECOND_ID, FAKE_SECRET)

    assert ident.resolve_antigravity_client([p], allow_env=False, verifier=verifier) == (
        SECOND_ID,
        FAKE_SECRET,
    )
    assert probed[0] == (FAKE_ID, SECOND_SECRET)
    assert (SECOND_ID, FAKE_SECRET) in probed


def test_back_to_back_secrets_stay_separate(tmp_path):
    """Two adjacent GOCSPX literals must not be captured as one value."""
    p = _write(
        tmp_path / "agy-fake",
        FAKE_ID.encode() + b"\x00" + FAKE_SECRET.encode() + SECOND_SECRET.encode() + b"\x00",
    )
    probed = []

    def verifier(pair):
        probed.append(pair)
        return False

    # Every pairing is rejected, so nothing is returned -- but each candidate
    # must carry exactly one secret, never the two concatenated.
    assert ident.resolve_antigravity_client([p], allow_env=False, verifier=verifier) is None
    assert [secret for _, secret in probed] == [FAKE_SECRET, SECOND_SECRET]


def test_verifier_not_consulted_when_unambiguous(tmp_path):
    p = _write(tmp_path / "agy-fake", FAKE_ID.encode() + b"\x00" + FAKE_SECRET.encode())

    def verifier(pair):
        raise AssertionError("probe must not run for a single unambiguous pairing")

    assert ident.resolve_antigravity_client([p], allow_env=False, verifier=verifier) == (
        FAKE_ID,
        FAKE_SECRET,
    )


def test_verifier_failure_does_not_fall_back_to_a_guess(tmp_path):
    """An unprobeable pairing (offline) must not be guessed at.

    Returning the first candidate here is what produced a credential that
    authenticates as nothing, with the failure only visible at token
    exchange time.
    """
    p = _write(
        tmp_path / "agy-fake",
        FAKE_ID.encode() + b"\x00" + SECOND_ID.encode() + b"\x00"
        + SECOND_SECRET.encode() + FAKE_SECRET.encode(),
    )

    def verifier(pair):
        raise ConnectionError("offline")

    assert ident.resolve_antigravity_client([p], allow_env=False, verifier=verifier) is None
