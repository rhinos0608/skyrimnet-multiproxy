"""Resolve OAuth client identity from environment or installed CLIs on disk.

Read-only helper: extracts the OAuth client id + secret pair embedded in
the installed Antigravity (``agy``) CLI artifact.
Performs no login, no network I/O, never executes the CLI, never shells
out. There is no hardcoded literal fallback anywhere in this module.

Precedence per provider: environment variables first, then CLI-on-disk
extraction. A pair is returned only when a client-id shaped value AND a
``GOCSPX-`` secret are found in the same file; partial values yield None.

One file can embed several OAuth clients (the ``agy`` binary ships more
than one), and ids and secrets live in separate string blobs, so a
secret's position relative to an id says nothing about which id it
belongs to. Such a file yields every id/secret *candidate* rather than
one guessed pair -- the provider's token endpoint is the only authority
on which combination authenticates. Callers that can afford one probe
pass a ``verifier`` callable to select the pairing; without one the
first candidate is returned, which is a guess. When a verifier is given
but cannot answer -- offline, endpoint error -- the identity stays
unresolved rather than falling back to a guess.

Every failure mode (CLI absent, unreadable, no match, malformed) returns
None -- public entry points never raise. Nothing is ever logged with
credential values, only source, file path, and redacted status.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Callable, Iterator, Literal, Optional

logger = logging.getLogger("proxy")

ClientPair = tuple[str, str]
ClientSource = Literal["env", "cli-path", "none"]
# Provider-endpoint probe returning True only for a pairing that authenticates.
Verifier = Callable[[ClientPair], bool]

_ENV_VARS: dict[str, tuple[str, str]] = {
    "antigravity": ("ANTIGRAVITY_CLIENT_ID", "ANTIGRAVITY_CLIENT_SECRET"),
}

_CLIENT_ID_RE = re.compile(rb"\d+-[A-Za-z0-9_.-]+\.apps\.googleusercontent\.com")
_SECRET_RE = re.compile(rb"GOCSPX-[A-Za-z0-9_-]{8,}")

_CHUNK_SIZE = 1024 * 1024
_OVERLAP = 256
_MAX_BYTES_PER_FILE = 256 * 1024 * 1024
_MAX_FILES_PER_PROVIDER = 8
_MAX_IDS_PER_FILE = 8
_MAX_SECRETS_PER_FILE = 8


def _candidate_paths(binary: str, extra: list[str]) -> list[str]:
    """Ordered CLI artifact paths: which() first, then known locations."""
    seen: list[str] = []
    try:
        found = shutil.which(binary)
    except Exception:
        found = None
    if found:
        seen.append(found)
    for path in extra:
        try:
            expanded = os.path.expanduser(path)
        except Exception:
            continue
        if expanded not in seen:
            seen.append(expanded)
    return seen


def _antigravity_search_paths() -> list[str]:
    home = os.path.expanduser("~")
    return _candidate_paths(
        "agy",
        [
            os.path.join(home, ".local", "bin", "agy"),
            "/usr/local/bin/agy",
            os.path.join(home, ".antigravity", "bin", "agy"),
        ],
    )


def _clean(value: bytes | str) -> Optional[str]:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or any(c in text for c in (" ", "\n", "\r", "\t", '"', "'", "\\")):
        return None
    return text


def _split_secrets(window: bytes) -> Iterator[str]:
    """Yield single secret literals found in a window, in file order.

    One greedy match can span two literals when a CLI build stores two
    secrets back-to-back (the Antigravity ``agy`` binary does: consecutive
    ``GOCSPX-`` values with no separator), so the scan continues inside the
    match instead of skipping the second value.
    """
    for match in _SECRET_RE.finditer(window):
        raw = match.group(0)
        while raw:
            repeated = raw.find(b"GOCSPX-", len("GOCSPX-"))
            head = raw[:repeated] if repeated != -1 else raw
            value = _clean(head)
            if value:
                yield value
            raw = raw[len(head):]


def _scan_file_candidates(path: str) -> list[ClientPair]:
    """Return every id/secret pairing one file implies, in file order.

    Bounded by the per-file id/secret caps and the single-pass size cap,
    because this runs over multi-hundred-megabyte native binaries.
    """
    ids: list[str] = []
    secrets: list[str] = []
    try:
        if not os.path.isfile(path):
            return []
        try:
            size = os.path.getsize(path)
        except OSError:
            return []
        if size <= 0 or size > _MAX_BYTES_PER_FILE:
            return []
        tail = b""
        scanned = 0
        with open(path, "rb") as f:
            while True:
                try:
                    chunk = f.read(_CHUNK_SIZE)
                except OSError:
                    return []
                if not chunk:
                    break
                scanned += len(chunk)
                if scanned > _MAX_BYTES_PER_FILE:
                    break
                window = tail + chunk
                for match in _CLIENT_ID_RE.finditer(window):
                    value = _clean(match.group(0))
                    if value and value not in ids and len(ids) < _MAX_IDS_PER_FILE:
                        ids.append(value)
                if len(secrets) < _MAX_SECRETS_PER_FILE:
                    for value in _split_secrets(window):
                        if value not in secrets:
                            secrets.append(value)
                        if len(secrets) >= _MAX_SECRETS_PER_FILE:
                            break
                if len(ids) >= _MAX_IDS_PER_FILE and len(secrets) >= _MAX_SECRETS_PER_FILE:
                    break
                tail = window[-_OVERLAP:] if len(window) > _OVERLAP else window
        return [(client_id, secret) for client_id in ids for secret in secrets]
    except OSError:
        return []
    except Exception:
        return []


def _from_env(provider: str) -> Optional[ClientPair]:
    names = _ENV_VARS.get(provider)
    if not names:
        return None
    cid = os.environ.get(names[0], "").strip()
    secret = os.environ.get(names[1], "").strip()
    if cid and secret:
        return (cid, secret)
    return None


def _choose_candidate(
    candidates: list[ClientPair],
    verifier: Optional[Verifier],
) -> Optional[ClientPair]:
    """One pairing: unambiguous files as-is, ambiguous ones by probe."""
    if len(candidates) == 1 or verifier is None:
        return candidates[0]
    try:
        for pair in candidates:
            if verifier(pair):
                logger.debug("OAuth client identity verified against provider endpoint")
                return pair
    except Exception:
        # The probe could not answer (offline, endpoint error). Guessing here
        # is what produced a credential that authenticates as nothing, so the
        # caller is told the identity is unresolved instead.
        logger.debug("OAuth client identity verification unavailable")
        return None
    # Every pairing was rejected: the file carries no usable credential.
    return None


def _from_disk(
    paths: list[str],
    *,
    verifier: Optional[Verifier] = None,
) -> tuple[Optional[ClientPair], Optional[str]]:
    checked = 0
    for path in paths:
        if checked >= _MAX_FILES_PER_PROVIDER:
            break
        try:
            if not os.path.isfile(path):
                continue
        except Exception:
            continue
        checked += 1
        try:
            candidates = _scan_file_candidates(path)
        except Exception:
            candidates = []
        if not candidates:
            continue
        pair = _choose_candidate(candidates, verifier)
        if pair:
            logger.debug("OAuth client identity found via cli-path")
            return pair, path
    return None, None


def _resolve_with_paths(
    provider: str,
    paths: list[str],
    *,
    allow_env: bool = True,
    verifier: Optional[Verifier] = None,
) -> tuple[Optional[ClientPair], ClientSource, Optional[str]]:
    try:
        if allow_env:
            pair = _from_env(provider)
            if pair:
                logger.debug("OAuth client identity found via env")
                return pair, "env", None
        pair, won = _from_disk(paths, verifier=verifier)
        if pair and won:
            return pair, "cli-path", won
        return None, "none", None
    except Exception:
        logger.debug("OAuth client identity lookup failed")
        return None, "none", None


def resolve_antigravity_client(
    search_paths: Optional[list[str]] = None,
    *,
    allow_env: bool = True,
    verifier: Optional[Verifier] = None,
) -> Optional[ClientPair]:
    """Return Antigravity (id, secret) from env or installed CLI, else None."""
    try:
        paths = list(search_paths) if search_paths is not None else _antigravity_search_paths()
        pair, _, _ = _resolve_with_paths(
            "antigravity", paths, allow_env=allow_env, verifier=verifier
        )
        return pair
    except Exception:
        logger.debug("OAuth client identity lookup failed")
        return None


def _normalize_provider(provider: str) -> Optional[str]:
    try:
        key = provider.strip().lower()
    except Exception:
        return None
    aliases = {
        "antigravity": "antigravity",
        "agy": "antigravity",
    }
    return aliases.get(key)


def resolve(
    provider: str,
    search_paths: Optional[list[str]] = None,
    *,
    allow_env: bool = True,
    verifier: Optional[Verifier] = None,
) -> Optional[ClientPair]:
    """Dispatch to the provider resolver; None on unknown provider/failure."""
    try:
        key = _normalize_provider(provider)
        if key == "antigravity":
            return resolve_antigravity_client(
                search_paths, allow_env=allow_env, verifier=verifier
            )
        return None
    except Exception:
        logger.debug("OAuth client identity lookup failed")
        return None


def client_source(
    provider: str,
    search_paths: Optional[list[str]] = None,
    *,
    allow_env: bool = True,
) -> ClientSource:
    """Report which source would win (env vs cli-path vs none), no values."""
    try:
        key = _normalize_provider(provider)
        if key is None:
            return "none"
        if allow_env and _from_env(key) is not None:
            return "env"
        paths = list(search_paths) if search_paths is not None else (
            _antigravity_search_paths()
        )
        pair, _ = _from_disk(paths)
        return "cli-path" if pair else "none"
    except Exception:
        return "none"


def resolve_with_source(
    provider: str,
    search_paths: Optional[list[str]] = None,
    *,
    allow_env: bool = True,
) -> tuple[Optional[ClientPair], ClientSource]:
    """Return (pair, source) without ever exposing values in logs."""
    try:
        pair = resolve(provider, search_paths, allow_env=allow_env)
        if pair is None:
            return None, "none"
        return pair, client_source(provider, search_paths, allow_env=allow_env)
    except Exception:
        return None, "none"
