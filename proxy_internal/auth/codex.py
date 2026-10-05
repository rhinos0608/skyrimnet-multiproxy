"""Codex OAuth token cache with refresh lock."""

import asyncio
import json
import logging
import os
import stat
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("proxy")

CODEX_CLI_VERSION = "0.155.1"
CODEX_USER_AGENT = f"codex_cli_rs/{CODEX_CLI_VERSION}"


class CodexAuthCache:
    """Stores OAuth tokens captured from Codex CLI auth flow."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.account_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_endpoint = "https://auth.openai.com/oauth/token"
        self._client_id = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI OAuth client ID
        # Lazy-initialised lock — see _get_lock(). Prevents concurrent refresh races.
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

    async def refresh_if_needed(self) -> bool:
        """Refresh access token if expired. Returns True if successful.

        First re-reads ~/.codex/auth.json (the CLI may have refreshed it
        since we loaded); only falls through to a network refresh when the
        on-disk tokens are still stale.
        """
        if not self.is_expired():
            return True
        # Prefer fresh on-disk tokens before spending a single-use refresh.
        try:
            if self.reload_from_disk_if_stale() and not self.is_expired():
                return True
        except Exception:
            pass
        if not self.refresh_token:
            logger.warning("Codex token expired but no refresh token available")
            return False

        async with self._get_lock():
            # Double-check: another coroutine may have refreshed while we waited
            if not self.is_expired():
                return True
            ok = await self._do_refresh()
            if ok:
                try:
                    self._persist_to_disk()
                except Exception as e:
                    logger.warning(f"Codex auth persist failed: {type(e).__name__}")
            return ok

    def get_chatgpt_headers(self) -> dict:
        """Full header set for POST {CHATGPT_CODEX_BASE_URL}/responses.

        Raises RuntimeError when no access token is cached. Never logs values.
        """
        import uuid

        if not self.access_token:
            raise RuntimeError("Codex OAuth not ready: no access token")
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=v1",
            "originator": "codex_cli_rs",
            "User-Agent": CODEX_USER_AGENT,
            "session_id": uuid.uuid4().hex,
            "x-client-request-id": str(uuid.uuid4()),
        }
        if self.account_id:
            headers["chatgpt-account-id"] = self.account_id
        return headers

    def reload_from_disk_if_stale(self) -> bool:
        """Re-read the CLI auth file and adopt its tokens when ours are stale.

        Returns True when the cache is ready afterwards (adopted fresh disk
        tokens or our own tokens were already usable). Returns False when
        both cache and disk lack a usable access token. Never logs values.
        """
        if self.access_token and not self.is_expired():
            return True
        path = _codex_auth_file_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return bool(self.access_token)
        if not isinstance(data, dict):
            return bool(self.access_token)
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return bool(self.access_token)
        disk_access = tokens.get("access_token")
        if not disk_access:
            return bool(self.access_token)
        disk_expiry = _expiry_from_disk_record(data)
        if disk_expiry is not None and disk_expiry <= datetime.now():
            return bool(self.access_token)
        # Adopt disk tokens (the CLI may have refreshed since we loaded).
        self.access_token = disk_access
        if tokens.get("refresh_token"):
            self.refresh_token = tokens["refresh_token"]
        if tokens.get("account_id"):
            self.account_id = tokens["account_id"]
        if disk_expiry is not None:
            self.expires_at = disk_expiry
        elif self.is_expired():
            self.expires_at = datetime.now() + timedelta(hours=23)
        return True

    def _persist_to_disk(self) -> None:
        """Atomically persist rotated tokens back to the CLI auth file.

        Preserves unknown top-level fields. temp file + os.replace, mode 0600.
        Never logs credential values or file contents.
        """
        path = _codex_auth_file_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            tokens = {}
            data["tokens"] = tokens
        if self.access_token:
            tokens["access_token"] = self.access_token
        if self.refresh_token:
            tokens["refresh_token"] = self.refresh_token
        if self.account_id:
            tokens["account_id"] = self.account_id
        data["last_refresh"] = datetime.now().astimezone().isoformat()
        fd, tmp = tempfile.mkstemp(dir=parent or ".", prefix=".auth.", suffix=".tmp")
        try:
            # mkstemp() already creates this 0600 (umask can only strip bits), but set
            # it explicitly before writing anything so the mode never depends on the
            # caller's umask, matching auth/antigravity.py's ordering.
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    async def _do_refresh(self) -> bool:
        """Perform the actual token refresh. Caller must hold self._refresh_lock."""
        logger.info("Refreshing Codex OAuth token...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self._client_id,
                }
                async with session.post(self._token_endpoint, json=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Token refresh failed: {resp.status} - {error[:200]}")
                        self.access_token = None
                        return False

                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    if data.get("refresh_token"):
                        self.refresh_token = data["refresh_token"]
                    expires_in = data.get("expires_in", 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    logger.info(f"Token refreshed, expires in {expires_in}s")
                    return True
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            self.access_token = None
            return False

    def apply_discovered_oauth(self, access_token: str, account_id: Optional[str] = None,
                               expires_in_s: int = 3300) -> bool:
        """Apply an on-disk-discovered OAuth token (no network, no logging of values).

        Only fills in what is missing: never overwrites an existing ready
        token, never touches the refresh token. Returns readiness after apply.
        """
        if self.is_ready:
            return True
        if not access_token or not isinstance(access_token, str):
            return False
        if self.access_token is None:
            self.access_token = access_token
        if account_id and not self.account_id:
            self.account_id = account_id
        if self.expires_at is None or self.is_expired():
            self.expires_at = datetime.now() + timedelta(seconds=expires_in_s)
        return self.is_ready

    def get_auth_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }


def _codex_auth_file_path() -> str:
    """Resolve the CLI auth file, honouring CODEX_HOME like the CLI does."""
    home_override = os.environ.get("CODEX_HOME")
    if home_override:
        return os.path.join(home_override, "auth.json")
    return os.path.expanduser("~/.codex/auth.json")


def _expiry_from_disk_record(data: dict) -> Optional[datetime]:
    """Derive token expiry from a CLI auth.json record (24h after refresh)."""
    last_refresh = data.get("last_refresh")
    if not last_refresh:
        return None
    try:
        dt = datetime.fromisoformat(str(last_refresh).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt + timedelta(hours=24)
    except (ValueError, TypeError):
        return None
