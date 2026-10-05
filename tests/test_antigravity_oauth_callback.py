"""Regression: Antigravity OAuth callback error pages must not 500.

The callback handler built its error responses with aiohttp's
``Response(..., status_code=...)``. aiohttp has no such keyword (it is
``status=``), so the TypeError escaped the handler and every error path --
including a rejected token exchange -- surfaced as an opaque HTTP 500 with
the real cause hidden. These tests drive the real callback server that the
login flow uses.

No provider network calls: only the pre-exchange validation paths, which
answer before any token request is made.
"""

from __future__ import annotations

import asyncio
import socket

import aiohttp
import pytest

import proxy


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(coro):
    """Run a coroutine, leaving the thread's current event loop as it was.

    ``asyncio.run`` clears the thread's event loop on exit, and
    ``tests/test_codex_direct_transport.py`` still resolves one through the
    legacy ``asyncio.get_event_loop()``. Restoring the previous loop keeps
    this module -- first by filename, so first to run -- from changing how
    the rest of the suite behaves.
    """
    policy = asyncio.get_event_loop_policy()
    try:
        previous = policy.get_event_loop()
    except RuntimeError:
        previous = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(
            previous if previous is not None and not previous.is_closed() else None
        )


def _get(port: int, query: str) -> tuple[int, str]:
    async def run() -> tuple[int, str]:
        saved = proxy._oauth_callback_server
        proxy._oauth_callback_server = None
        runner = await proxy.start_oauth_callback_server()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://127.0.0.1:{port}/oauth-callback?{query}"
                async with session.get(url, headers={"Referer": url}) as response:
                    return response.status, await response.text()
        finally:
            proxy._oauth_callback_server = saved
            await runner.cleanup()

    return _run(run())


@pytest.fixture()
def callback_port(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(proxy, "ANTIGRAVITY_OAUTH_PORT", port)
    return port


def test_missing_state_is_400_not_500(callback_port):
    status, body = _get(callback_port, "code=some-code")
    assert status == 400
    assert "Missing code or state" in body


def test_unknown_state_is_400_not_500(callback_port):
    status, body = _get(callback_port, "code=some-code&state=unknown-state")
    assert status == 400
    assert "Invalid state" in body


def test_callback_does_not_log_code_or_referer(callback_port, caplog):
    with caplog.at_level("INFO", logger="aiohttp.access"):
        status, _ = _get(callback_port, "code=fixture-private-code&state=unknown-state")
    assert status == 400
    assert "fixture-private-code" not in caplog.text
    assert not any(record.name == "aiohttp.access" for record in caplog.records)


def test_native_mode_rejects_browser_callback(callback_port, monkeypatch):
    monkeypatch.setattr(proxy.antigravity_auth, "native_keychain", True)
    status, body = _get(callback_port, "code=fixture-code&state=unknown-state")
    assert status == 409
    assert "agy" in body
