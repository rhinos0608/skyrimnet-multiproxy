"""Antigravity multi-account OAuth auth, encryption, and file persistence."""

import asyncio
import json
import logging
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from proxy_internal.encryption import _get_fernet, _encrypt_value, _decrypt_value, _ENC_PREFIX
from proxy_internal.credential_discovery import native_antigravity_enabled, discover_native_antigravity_token

logger = logging.getLogger("proxy")

class _AntigravityPersistenceCoordinator:
    """Order auth snapshots so older workers cannot replace newer state."""

    def __init__(self):
        self._gen_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._next_generation = 0
        self._latest_generation = -1

    def capture(self, snapshot_factory):
        """Assign generation while taking snapshot under generation lock."""
        with self._gen_lock:
            generation = self._next_generation
            self._next_generation += 1
            return generation, snapshot_factory()

    def next_generation(self) -> int:
        with self._gen_lock:
            generation = self._next_generation
            self._next_generation += 1
            return generation

    def replace(self, generation: int, writer) -> bool:
        """Replace file only when snapshot is not older than committed state."""
        with self._commit_lock:
            if generation < self._latest_generation:
                return False
            writer()
            self._latest_generation = generation
            return True


_ANTIGRAVITY_PERSISTENCE = _AntigravityPersistenceCoordinator()

# Antigravity auth-file encryption field set.  Previously lived in proxy.py
# as `_ENCRYPTED_AUTH_FIELDS`; moved here because it is only used by the
# encrypt/decrypt helpers below.
_ENCRYPTED_AUTH_FIELDS = {"refresh_token", "access_token"}


# Canonical wire fingerprint from the official Antigravity client
# (see pi-provider-antigravity constants.ts): fixed UA with aidev_client
# metadata. Headers are exactly Host (set by aiohttp from the URL),
# User-Agent, Content-Type, Authorization — no Client-Metadata or
# X-Goog-Api-Client extras.
_ANTIGRAVITY_USER_AGENT = (
    "antigravity/cli/1.2.12 (aidev_client; os_type=linux; arch=amd64; "
    "cl=989022288; auth_method=consumer)"
)


def get_antigravity_headers() -> dict:
    """Get headers for Antigravity API requests."""
    return {
        "User-Agent": _ANTIGRAVITY_USER_AGENT,
    }


# Token endpoint probe.  The official CLI binary embeds more than one OAuth
# client, and ids and secrets are separate string blobs whose order carries no
# pairing information, so the pairing has to be settled by the authority that
# actually issues tokens.
_OAUTH_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_OAUTH_PROBE_TIMEOUT_S = 6.0


def verify_antigravity_client_pair(pair: tuple, redirect_uri: str) -> bool:
    """True only when Google accepts this (client_id, client_secret) pair.

    Probes the token endpoint with a deliberately bogus authorization code:
    per RFC 6749 section 5.2 ``invalid_client`` means client authentication
    failed, while ``invalid_grant`` means the credential itself was accepted
    and only the grant was rejected.  ``redirect_uri`` must match the one the
    login flow uses, or the endpoint reports a URI mismatch instead of
    validating the credential.

    Raises when the probe cannot be completed (no network, endpoint error), so
    an unresolved credential is never mistaken for a rejected one; the
    credential itself is never logged.
    """
    if not pair or not pair[0] or not pair[1]:
        return False
    payload = {
        "client_id": pair[0],
        "client_secret": pair[1],
        "grant_type": "authorization_code",
        "code": "antigravity-client-identity-probe",
        "redirect_uri": redirect_uri,
        "code_verifier": "x" * 128,
    }
    request = urllib.request.Request(
        _OAUTH_TOKEN_ENDPOINT,
        data=urllib.parse.urlencode(payload).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_OAUTH_PROBE_TIMEOUT_S) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    try:
        return json.loads(body).get("error") == "invalid_grant"
    except Exception:
        return False


class AntigravityAccount:
    """Stores OAuth tokens for a single Google Antigravity account."""

    def __init__(self):
        self.native_keychain = False
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.email: Optional[str] = None
        self.project_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._error_count: int = 0  # Track consecutive errors for this account
        # Lazy-initialised lock — see _get_lock(). Prevents concurrent refresh races
        # across coroutines sharing this account.
        self._refresh_lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-init the refresh lock so it binds to the running event loop."""
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    @property
    def is_ready(self) -> bool:
        return self.access_token is not None and not self.is_expired()

    def is_expired(self) -> bool:
        if not self.expires_at:
            return True
        return datetime.now() >= self.expires_at - timedelta(minutes=5)  # 5min buffer

    def to_dict(self) -> dict:
        """Serialize account to dict for saving."""
        return {
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "email": self.email,
            "project_id": self.project_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AntigravityAccount":
        """Create account from dict."""
        account = cls()
        account.refresh_token = data.get("refresh_token")
        account.email = data.get("email")
        account.project_id = data.get("project_id")
        if data.get("access_token"):
            account.access_token = data["access_token"]
            if data.get("expires_at"):
                try:
                    account.expires_at = datetime.fromisoformat(data["expires_at"])
                except ValueError:
                    account.expires_at = datetime.now() - timedelta(seconds=1)
            else:
                account.expires_at = datetime.now() - timedelta(seconds=1)
        return account

    def reload_native(self) -> bool:
        """Re-read CLI-owned access token; never mint or persist credentials."""
        token = discover_native_antigravity_token()
        previous = self.access_token
        self.access_token = token.access_token if token else None
        self.expires_at = datetime.fromtimestamp(token.expires_at.timestamp()) if token else None
        self.refresh_token = None
        if self.access_token != previous:
            self._error_count = 0
        return self.is_ready

    async def refresh_if_needed(self) -> bool:
        """Refresh owned tokens, or re-read agy's session without refreshing it."""
        if self.native_keychain:
            async with self._get_lock():
                return await asyncio.to_thread(self.reload_native)
        if not self.is_expired():
            return True
        if not self.refresh_token:
            logger.warning(f"Antigravity token expired for {self.email} but no refresh token available")
            return False

        async with self._get_lock():
            # Double-check: another coroutine may have refreshed while we waited
            if not self.is_expired():
                return True
            return await self._do_refresh()

    async def _do_refresh(self) -> bool:
        """Perform the actual token refresh. Caller must hold self._refresh_lock."""
        if self.native_keychain:
            return await asyncio.to_thread(self.reload_native)
        import proxy
        logger.info(f"Refreshing Antigravity OAuth token for {self.email}...")
        oauth_client = proxy.get_antigravity_oauth_client()
        if oauth_client is None:
            logger.error(proxy.antigravity_oauth_unresolved_error())
            self.access_token = None
            return False
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": oauth_client[0],
                    "client_secret": oauth_client[1],
                }
                async with session.post("https://oauth2.googleapis.com/token", data=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Antigravity token refresh failed for {self.email}: {resp.status} - {error[:200]}")
                        self.access_token = None
                        return False

                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    if data.get("refresh_token"):
                        self.refresh_token = data["refresh_token"]
                    expires_in = data.get("expires_in", 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    logger.info(f"Antigravity token refreshed for {self.email}, expires in {expires_in}s")
                    return True
        except Exception as e:
            logger.error(f"Antigravity token refresh error for {self.email}: {e}")
            self.access_token = None
            return False

    def get_auth_headers(self) -> dict:
        """Get headers for API requests."""
        headers = get_antigravity_headers()
        headers["Authorization"] = f"Bearer {self.access_token}"
        headers["Content-Type"] = "application/json"
        return headers


class AntigravityAuthCache:
    """Manages multiple Google Antigravity accounts with automatic fallback."""

    def __init__(self):
        self.accounts: list[AntigravityAccount] = []
        self._current_index: int = 0
        # Legacy single-account properties for backward compatibility
        self._legacy_account = AntigravityAccount()
        self._persist_lock: Optional[asyncio.Lock] = None
        self.native_keychain = native_antigravity_enabled()
        if self.native_keychain:
            account = AntigravityAccount()
            account.native_keychain = True
            account.email = "agy-keychain"
            self.accounts.append(account)

    def reload_native(self) -> bool:
        """Re-read the sole native session; never fall back to proxy OAuth."""
        return self.accounts[0].reload_native()

    def _get_persist_lock(self) -> asyncio.Lock:
        if self._persist_lock is None:
            self._persist_lock = asyncio.Lock()
        return self._persist_lock

    def _get_active_accounts(self) -> list[AntigravityAccount]:
        """Get list of accounts sorted by error count (prefer healthy accounts)."""
        if not self.accounts:
            return [self._legacy_account] if self._legacy_account.refresh_token else []
        # Sort by error count, but preserve order for accounts with same error count
        return sorted(self.accounts, key=lambda a: a._error_count)

    def get_next_account(self) -> Optional[AntigravityAccount]:
        """Get next available account using round-robin with error-aware prioritization."""
        active = self._get_active_accounts()
        if not active:
            return None
        # Pick the account with lowest error count, then round-robin among equals
        min_errors = min(a._error_count for a in active)
        good_accounts = [a for a in active if a._error_count == min_errors]
        if len(good_accounts) == 1:
            return good_accounts[0]
        # Round-robin among good accounts
        account = good_accounts[self._current_index % len(good_accounts)]
        self._current_index += 1
        return account

    def mark_account_error(self, account: AntigravityAccount, status_code: int):
        """Mark an account as having an error (increments error count)."""
        account._error_count += 1
        logger.warning(f"Antigravity account {account.email} got {status_code} error (error count: {account._error_count})")

    def mark_account_success(self, account: AntigravityAccount):
        """Mark an account as successful (resets error count)."""
        if account._error_count > 0:
            logger.info(f"Antigravity account {account.email} recovered, resetting error count")
        account._error_count = 0

    @property
    def is_ready(self) -> bool:
        """Check if any account is ready."""
        if self.accounts:
            return any(a.is_ready for a in self.accounts)
        return self._legacy_account.is_ready

    def is_expired(self) -> bool:
        """Check if current account is expired (for backward compatibility)."""
        account = self.get_next_account()
        return account.is_expired() if account else True

    @property
    def email(self) -> Optional[str]:
        """Get email of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.email
        if self._legacy_account.email:
            return self._legacy_account.email
        return self.accounts[0].email if self.accounts else None

    @property
    def project_id(self) -> Optional[str]:
        """Get project_id of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.project_id
        if self._legacy_account.project_id:
            return self._legacy_account.project_id
        return self.accounts[0].project_id if self.accounts else None

    @property
    def session(self) -> Optional[aiohttp.ClientSession]:
        """Get session of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready and account.session:
                return account.session
        return self._legacy_account.session

    async def refresh_if_needed(self) -> bool:
        """Refresh tokens for all accounts. Returns True if at least one is ready.

        Snapshots ``self.accounts`` at entry so a concurrent
        ``add_account``/``remove_account`` cannot mutate the list mid-loop
        (the per-account ``await`` yields the event loop between iterations).
        The final readiness check also uses the snapshot so the concurrency
        window is fully closed.

        If any account's tokens changed during refresh, persists to disk
        atomically so refreshed tokens are not lost on restart.
        """
        if self.native_keychain:
            return await self.accounts[0].refresh_if_needed()
        import proxy

        def _token_snapshot(acct):
            return (acct.access_token, acct.refresh_token,
                    acct.expires_at.isoformat() if acct.expires_at else None)

        dirty = False

        if self.accounts:
            accounts_snapshot = list(self.accounts)
            before = [_token_snapshot(a) for a in accounts_snapshot]
            for account in accounts_snapshot:
                if account.refresh_token:
                    await account.refresh_if_needed()
                    if account.is_ready and not account.session:
                        account.session = proxy.create_session()
            after = [_token_snapshot(a) for a in accounts_snapshot]
            if before != after:
                dirty = True
            result = any(a.is_ready for a in accounts_snapshot)
        else:
            # Legacy single-account mode
            before = _token_snapshot(self._legacy_account)
            result = await self._legacy_account.refresh_if_needed()
            after = _token_snapshot(self._legacy_account)
            if before != after:
                dirty = True

        if dirty:
            async with self._get_persist_lock():
                generation, accounts_snapshot = _ANTIGRAVITY_PERSISTENCE.capture(
                    lambda: _snapshot_antigravity_auth_accounts(self)
                )
                persist_task = asyncio.create_task(
                    asyncio.to_thread(
                        _write_antigravity_auth_snapshot,
                        accounts_snapshot,
                        generation,
                    )
                )
                try:
                    await asyncio.shield(persist_task)
                except asyncio.CancelledError:
                    await asyncio.shield(persist_task)
                    raise
                except Exception:
                    logger.warning(
                        "Antigravity auth persistence failed", exc_info=True
                    )

        return result

    def get_auth_headers(self) -> dict:
        """Get auth headers for current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.get_auth_headers()
        return self._legacy_account.get_auth_headers()

    def add_account(self, account: AntigravityAccount):
        """Add a proxy-owned account to the legacy pool."""
        if self.native_keychain:
            raise RuntimeError("Antigravity uses the agy Keychain session; manage accounts in agy")
        # Check if account with this email already exists
        for i, existing in enumerate(self.accounts):
            if existing.email == account.email:
                # Update existing account
                self.accounts[i] = account
                logger.info(f"Updated Antigravity account: {account.email}")
                return
        self.accounts.append(account)
        logger.info(f"Added Antigravity account: {account.email}")

    def remove_account(self, email: str) -> bool:
        """Remove a proxy-owned account by email. Native login belongs to agy."""
        if self.native_keychain:
            return False
        for i, account in enumerate(self.accounts):
            if account.email == email:
                if account.session:
                    # Schedule session cleanup
                    asyncio.create_task(account.session.close())
                del self.accounts[i]
                logger.info(f"Removed Antigravity account: {email}")
                return True
        return False

    def get_all_accounts_info(self) -> list[dict]:
        """Get info about all accounts for display."""
        result = []
        for account in self.accounts:
            result.append({
                "email": account.email,
                "project_id": account.project_id,
                "is_ready": account.is_ready,
                "error_count": account._error_count,
            })
        # Include legacy account if it exists and not already in list
        if self._legacy_account.refresh_token:
            legacy_email = self._legacy_account.email
            if not any(a["email"] == legacy_email for a in result):
                result.insert(0, {
                    "email": legacy_email,
                    "project_id": self._legacy_account.project_id,
                    "is_ready": self._legacy_account.is_ready,
                    "error_count": self._legacy_account._error_count,
                })
        return result


def _decrypt_auth_account(account_data: dict) -> dict:
    """Decrypt encrypted fields in an antigravity account dict. Auto-migrates plaintext."""
    fernet = _get_fernet()
    out = dict(account_data)
    for field in _ENCRYPTED_AUTH_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val.startswith(_ENC_PREFIX):
            out[field] = _decrypt_value(fernet, val)
    return out


def _encrypt_auth_account(account_data: dict) -> dict:
    """Encrypt sensitive fields in an antigravity account dict before saving."""
    fernet = _get_fernet()
    out = dict(account_data)
    for field in _ENCRYPTED_AUTH_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
            out[field] = _encrypt_value(fernet, val)
    return out


async def load_antigravity_auth():
    """Load Antigravity auth from cached file. Supports both legacy single-account and new multi-account formats.

    Recapture order: (1) re-read own on-disk store (it may have been
    refreshed by its CLI since load), (2) adopt a matching-provider Pi
    ``antigravity`` OAuth record, (3) caller surfaces the interactive
    ``/config/antigravity-login`` flow as last resort. Never adopts another
    provider's credential; expired records never count as fresh.
    """
    import proxy

    if proxy.antigravity_auth.native_keychain:
        ready = await proxy.antigravity_auth.refresh_if_needed()
        logger.info("Antigravity native agy session %s", "ready via Keychain" if ready else "unavailable -- sign in with agy")
        return

    if not os.path.exists(proxy.ANTIGRAVITY_AUTH_FILE):
        if _adopt_discovered_antigravity_token(proxy):
            logger.info("Antigravity auth recaptured via discovery (redacted)")
        else:
            logger.info("No Antigravity auth file found -- visit /config/antigravity-login")
        return

    await _load_antigravity_file(proxy)
    if proxy.antigravity_auth.is_ready:
        return
    if _adopt_discovered_antigravity_token(proxy):
        logger.info("Antigravity auth recaptured via discovery (redacted)")


ANTIGRAVITY_DISCOVERY_EMAIL = "pi-autodiscovery"


def _adopt_discovered_antigravity_token(proxy_module=None) -> bool:
    """Adopt matching-provider Pi ``antigravity`` OAuth record. No network, no values logged."""
    import proxy as _proxy_default
    proxy_module = proxy_module or _proxy_default
    if proxy_module.antigravity_auth.native_keychain:
        return proxy_module.antigravity_auth.reload_native()
    try:
        from proxy_internal.credential_discovery import discover_antigravity_oauth
        token = discover_antigravity_oauth()
    except Exception:
        return False
    if not token:
        return False
    return adopt_discovered_antigravity_token(token, proxy_module=proxy_module)


def _unwrap_discovered_token(token: str) -> tuple[str, Optional[str]]:
    """Unwrap a discovered Pi ``antigravity`` OAuth record value.

    Pi stores the record's ``access`` either as a bare bearer token or as
    a JSON blob ``{"token": ..., "projectId": ...}``. The blob itself
    is never a valid Authorization value (sending it yields 401), so
    extract the nested token and project. Returns (token, project_id);
    project_id is None when the value carries none. Unparseable or empty
    values yield ("", None) so the caller rejects them.
    """
    if not isinstance(token, str) or not token.strip():
        return "", None
    text = token.strip()
    if not text.startswith("{"):
        return text, None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return "", None
    if not isinstance(obj, dict):
        return "", None
    nested = obj.get("token")
    project = obj.get("projectId")
    if not isinstance(nested, str) or not nested.strip():
        return "", None
    return nested.strip(), project.strip() if isinstance(project, str) and project.strip() else None


def adopt_discovered_antigravity_token(token: str, expires_in_seconds: int = 3300, proxy_module=None) -> bool:
    """Attach a discovered matching-provider token as a ready account."""
    import proxy as _proxy_default
    from datetime import datetime as _dt, timedelta as _td
    proxy_module = proxy_module or _proxy_default
    if proxy_module.antigravity_auth.native_keychain:
        return False
    bare, project_id = _unwrap_discovered_token(token)
    if not bare:
        return False
    cache = proxy_module.antigravity_auth
    for existing in cache.accounts:
        if existing.email == ANTIGRAVITY_DISCOVERY_EMAIL:
            existing.access_token = bare
            if project_id and not existing.project_id:
                existing.project_id = project_id
            if existing.is_expired():
                existing.expires_at = _dt.now() + _td(seconds=expires_in_seconds)
            return existing.is_ready
    account = AntigravityAccount()
    account.access_token = bare
    account.email = ANTIGRAVITY_DISCOVERY_EMAIL
    if project_id:
        account.project_id = project_id
    account.expires_at = _dt.now() + _td(seconds=expires_in_seconds)
    cache.add_account(account)
    return cache.is_ready


def resync_antigravity_from_disk(proxy_module=None) -> bool:
    """Sync re-read of own auth file; adopts only unexpired access tokens. No network."""
    import proxy as _proxy_default
    proxy_module = proxy_module or _proxy_default
    if proxy_module.antigravity_auth.native_keychain:
        return proxy_module.antigravity_auth.reload_native()
    path = proxy_module.ANTIGRAVITY_AUTH_FILE
    if not os.path.exists(path):
        return proxy_module.antigravity_auth.is_ready
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return proxy_module.antigravity_auth.is_ready
    entries = data.get("accounts") if isinstance(data, dict) and isinstance(data.get("accounts"), list) else [data]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            account = AntigravityAccount.from_dict(_decrypt_auth_account(entry))
        except Exception:
            continue
        if account.access_token and not account.is_expired():
            if account.refresh_token:
                proxy_module.antigravity_auth.add_account(account)
            else:
                adopt_discovered_antigravity_token(account.access_token, proxy_module=proxy_module)
    return proxy_module.antigravity_auth.is_ready


async def recapture_antigravity_auth(proxy_module=None) -> Optional[str]:
    """Recapture stale auth: file first, then matching discovery. Returns source or None."""
    import proxy as _proxy_default
    proxy_module = proxy_module or _proxy_default
    if proxy_module.antigravity_auth.native_keychain:
        return "keychain" if await proxy_module.antigravity_auth.refresh_if_needed() else None
    if proxy_module.antigravity_auth.is_ready:
        return "ready"
    if os.path.exists(proxy_module.ANTIGRAVITY_AUTH_FILE):
        await _load_antigravity_file(proxy_module)
        if proxy_module.antigravity_auth.is_ready:
            logger.info("Antigravity auth recaptured via file (redacted)")
            return "file"
    if _adopt_discovered_antigravity_token(proxy_module):
        logger.info("Antigravity auth recaptured via discovery (redacted)")
        return "discovery"
    return None


async def _load_antigravity_file(proxy):
    """Read own auth store (file may have been refreshed externally)."""
    if proxy.antigravity_auth.native_keychain:
        return
    try:
        with open(proxy.ANTIGRAVITY_AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        migrated = False
        # Check if it's the new multi-account format (has "accounts" array)
        if "accounts" in data and isinstance(data["accounts"], list):
            # New multi-account format
            for account_data in data["accounts"]:
                decrypted = _decrypt_auth_account(account_data)
                # Check if any field was plaintext (needs migration)
                for field in _ENCRYPTED_AUTH_FIELDS:
                    val = account_data.get(field, "")
                    if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
                        migrated = True
                account = AntigravityAccount.from_dict(decrypted)
                if account.refresh_token:
                    if await account.refresh_if_needed():
                        account.session = proxy.create_session()
                        proxy.antigravity_auth.add_account(account)
                        logger.info(f"Antigravity auth loaded for {account.email}")
                    else:
                        # Still add the account even if refresh failed - it might work later
                        proxy.antigravity_auth.add_account(account)
                        logger.warning(f"Antigravity token refresh failed for {account.email} -- may need re-login")
            logger.info(f"Loaded {len(proxy.antigravity_auth.accounts)} Antigravity account(s)")
        else:
            # Legacy single-account format
            decrypted = _decrypt_auth_account(data)
            for field in _ENCRYPTED_AUTH_FIELDS:
                val = data.get(field, "")
                if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
                    migrated = True
            account = AntigravityAccount.from_dict(decrypted)
            if account.refresh_token:
                if await account.refresh_if_needed():
                    account.session = proxy.create_session()
                    proxy.antigravity_auth._legacy_account = account
                    logger.info(f"Antigravity auth loaded for {account.email} (legacy mode)")
                else:
                    proxy.antigravity_auth._legacy_account = account
                    logger.warning("Antigravity token refresh failed -- re-login required")
        if migrated:
            _save_antigravity_auth()
            logger.info("Migrated antigravity-auth.json tokens to encrypted storage")
    except Exception as e:
        logger.error(f"Failed to load Antigravity auth: {e}")


def _snapshot_antigravity_auth_accounts(
    auth_cache: Optional[AntigravityAuthCache] = None,
) -> list[dict]:
    """Copy mutable account state before persistence leaves the event loop."""
    if auth_cache is None:
        import proxy
        auth_cache = proxy.antigravity_auth

    if auth_cache.native_keychain:
        return []
    accounts_data = []
    legacy = auth_cache._legacy_account
    if legacy.refresh_token:
        accounts_data.append(legacy.to_dict())
    for account in auth_cache.accounts:
        if account.native_keychain:
            continue
        if not legacy.refresh_token or account.email != legacy.email:
            accounts_data.append(account.to_dict())
    return accounts_data


def _write_antigravity_auth_snapshot(
    accounts_data: list[dict], generation: Optional[int] = None
) -> None:
    """Encrypt and atomically persist an immutable account snapshot."""
    import proxy

    if generation is None:
        generation = _ANTIGRAVITY_PERSISTENCE.next_generation()
    encrypted = [_encrypt_auth_account(account) for account in accounts_data]
    payload = encrypted[0] if len(encrypted) == 1 else {"accounts": encrypted}
    _ANTIGRAVITY_PERSISTENCE.replace(
        generation,
        lambda: _write_antigravity_auth_file(proxy.ANTIGRAVITY_AUTH_FILE, payload),
    )


def _write_antigravity_auth_file(target: str, payload: dict) -> None:
    """Atomically write one payload while the caller holds persistence lock."""
    target_dir = os.path.dirname(target) or "."
    target_name = os.path.basename(target)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target_name}.", suffix=".tmp", dir=target_dir
    )
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            tmp_fd = None
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
        # Ensure the rename is durable before reporting completion.
        try:
            dir_fd = os.open(target_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        tmp_path = None
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _save_antigravity_auth():
    """Synchronously persist proxy-owned auth only; never overwrite with native state."""
    import proxy
    if proxy.antigravity_auth.native_keychain:
        return
    generation, accounts_snapshot = _ANTIGRAVITY_PERSISTENCE.capture(
        lambda: _snapshot_antigravity_auth_accounts()
    )
    _write_antigravity_auth_snapshot(accounts_snapshot, generation)
