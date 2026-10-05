"""Per-provider concurrency pools: limits, queueing, telemetry, and wiring.

Every upstream is faked — no network, no live credentials.
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proxy
from proxy_internal import concurrency, provider_config


@pytest.fixture(autouse=True)
def clean_pools():
    concurrency.reset()
    yield
    concurrency.reset()


def _set_provider(monkeypatch, provider, **settings):
    """Inject a config.json `providers.<name>` entry (restored after the test)."""
    monkeypatch.setitem(provider_config._cache, provider, settings)


@asynccontextmanager
async def _asgi_client():
    """In-loop HTTP client for the app (no lifespan: no auth capture, no CLI)."""
    transport = httpx.ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class TestLimitResolution:
    def test_zai_defaults_to_its_plan_cap(self):
        assert concurrency._limit_for("zai") == 6

    def test_other_providers_unlimited_by_default(self):
        assert concurrency._limit_for("openrouter") == 0

    def test_config_overrides_the_default(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=2)
        assert concurrency._limit_for("zai") == 2

    def test_zero_disables_the_limit(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=0)
        assert concurrency._limit_for("zai") == 0
        assert concurrency.snapshot() == {}

    def test_patch_validation_rejects_and_clamps(self):
        for bad in ({"max_concurrent": True}, {"queue_timeout_ms": "soon"}):
            with pytest.raises(ValueError):
                provider_config.validate_patch(bad)
        assert provider_config.validate_patch({"max_concurrent": 500})["max_concurrent"] == 64
        assert provider_config.validate_patch({"queue_timeout_ms": -1})["queue_timeout_ms"] == 0


class TestSlotAcquisition:
    @pytest.mark.asyncio
    async def test_unlimited_provider_yields_an_inert_slot(self):
        slot = await concurrency.acquire("openrouter", request_id="t1")
        slot.release()
        slot.release()  # idempotent
        assert "openrouter" not in concurrency.snapshot()

    @pytest.mark.asyncio
    async def test_waits_for_a_slot_freed_by_another_request(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=1, queue_timeout_ms=5000)
        held = await concurrency.acquire("zai", request_id="t1")

        waiting = asyncio.create_task(concurrency.acquire("zai", request_id="t2"))
        await asyncio.sleep(0)
        assert concurrency.snapshot()["zai"]["queued"] == 1
        assert not waiting.done()

        held.release()
        second = await asyncio.wait_for(waiting, timeout=1)
        assert concurrency.snapshot()["zai"]["active"] == 1
        second.release()
        assert concurrency.snapshot()["zai"]["active"] == 0

    @pytest.mark.asyncio
    async def test_queue_timeout_surfaces_as_503(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=1, queue_timeout_ms=30)
        held = await concurrency.acquire("zai", request_id="t1")

        with pytest.raises(HTTPException) as excinfo:
            await concurrency.acquire("zai", request_id="t2")
        assert excinfo.value.status_code == 503
        assert "zai" in excinfo.value.detail

        held.release()
        stats = concurrency.snapshot()["zai"]
        assert stats["timed_out_total"] == 1
        assert stats["active"] == 0
        assert stats["queued"] == 0

    @pytest.mark.asyncio
    async def test_raised_limit_applies_once_the_pool_is_idle(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=1, queue_timeout_ms=30)
        held = await concurrency.acquire("zai", request_id="t1")

        # A live pool with a holder keeps its original limit: swapping it now
        # would over-admit callers already inside.
        _set_provider(monkeypatch, "zai", max_concurrent=2, queue_timeout_ms=200)
        with pytest.raises(HTTPException):
            await concurrency.acquire("zai", request_id="t2")

        held.release()
        first = await concurrency.acquire("zai", request_id="t3")
        second = await concurrency.acquire("zai", request_id="t4")
        assert concurrency.snapshot()["zai"]["active"] == 2
        first.release()
        second.release()

    @pytest.mark.asyncio
    async def test_hold_slot_releases_when_the_stream_closes(self, monkeypatch):
        _set_provider(monkeypatch, "zai", max_concurrent=1, queue_timeout_ms=1000)
        slot = await concurrency.acquire("zai", request_id="t1")

        async def _events():
            yield "one"
            yield "two"

        stream = concurrency.hold_slot(_events(), slot)
        assert await stream.__anext__() == "one"
        assert concurrency.snapshot()["zai"]["active"] == 1

        await stream.aclose()
        assert concurrency.snapshot()["zai"]["active"] == 0

        # The slot really is free, not just uncounted.
        replacement = await concurrency.acquire("zai", request_id="t2")
        replacement.release()


class TestHealthTelemetry:
    def test_health_reports_pools_including_idle_ones(self, test_client, monkeypatch):
        # /health probes provider discovery, which can attach a live account to the
        # shared auth caches. This test is about the concurrency block only, so
        # keep discovery inert rather than leaking state into later tests.
        monkeypatch.setattr(proxy, "ensure_antigravity_discovery", lambda: False, raising=False)
        monkeypatch.setattr(proxy, "ensure_codex_discovery", lambda: False, raising=False)

        body = test_client.get("/health").json()
        assert body["concurrency"]["zai"]["limit"] == 6
        assert body["concurrency"]["zai"]["active"] == 0


class TestChatEndpointWiring:
    """The chat path must name the provider whose pool it takes a slot from."""

    @staticmethod
    def _body(model="zai:glm-4.5"):
        return {"model": model, "messages": [{"role": "user", "content": "hi"}]}

    @pytest.mark.asyncio
    async def test_full_pool_returns_503_then_the_call_succeeds(self, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy, "zai_api_key", "k-test", raising=False)
        _set_provider(monkeypatch, "zai", max_concurrent=1, queue_timeout_ms=30)

        calls = []

        async def _fake_direct(system_prompt, messages, model, max_tokens, **extra):
            calls.append(model)
            return "hello"

        monkeypatch.setattr(proxy, "call_zai_direct", _fake_direct)

        held = await concurrency.acquire("zai", request_id="holder")
        async with _asgi_client() as client:
            busy = await client.post("/v1/chat/completions", json=self._body())
            assert busy.status_code == 503
            assert "zai" in busy.json()["detail"]
            assert calls == []  # never reached the provider

            held.release()
            ok = await client.post("/v1/chat/completions", json=self._body())
            assert ok.status_code == 200
            assert ok.json()["choices"][0]["message"]["content"] == "hello"

        assert len(calls) == 1
        assert concurrency.snapshot()["zai"]["active"] == 0  # slot returned

    @pytest.mark.asyncio
    async def test_streaming_holds_the_slot_until_the_stream_ends(self, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy, "zai_api_key", "k-test", raising=False)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _fake_stream(system_prompt, messages, model, max_tokens, **extra):
            yield 'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            started.set()
            await release.wait()
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(proxy, "call_zai_streaming", _fake_stream)
        body = self._body()
        body["stream"] = True

        async def _request():
            async with _asgi_client() as client:
                return await client.post("/v1/chat/completions", json=body)

        pending = asyncio.create_task(_request())
        await asyncio.wait_for(started.wait(), timeout=5)
        # Mid-stream the provider's slot is still held...
        assert concurrency.snapshot()["zai"]["active"] == 1

        release.set()
        response = await asyncio.wait_for(pending, timeout=5)
        assert response.status_code == 200
        assert "hi" in response.text
        # ...and released once the stream is done.
        assert concurrency.snapshot()["zai"]["active"] == 0

    @pytest.mark.asyncio
    async def test_unlimited_providers_never_touch_a_pool(self, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy, "zai_api_key", "k-test", raising=False)

        async def _fake_direct(system_prompt, messages, model, max_tokens, **extra):
            return "hello"

        monkeypatch.setattr(proxy, "call_zai_direct", _fake_direct)
        async with _asgi_client() as client:
            ok = await client.post("/v1/chat/completions", json=self._body("zai:glm-4.5"))
            assert ok.status_code == 200
        # zai keeps its 6-call default, so a pool exists — but this request alone
        # must leave it idle rather than holding a slot.
        assert concurrency.snapshot()["zai"]["active"] == 0
