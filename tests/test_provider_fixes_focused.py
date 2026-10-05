"""Focused tests for CommandCode/Ollama/LMStudio provider fixes."""

from proxy_internal.providers.lmstudio import (
    _fetch_lmstudio_models,
    _lmstudio_base_url,
    _resolve_lmstudio_model,
)
from proxy_internal.providers.ollama import _fetch_ollama_models, _resolve_ollama_model


def test_command_code_passthrough_unlisted(monkeypatch):
    from proxy_internal.providers import command_code as cc

    assert cc._cli_fallback_target("vendor/unlisted-model-xyz") == "vendor/unlisted-model-xyz"
    assert cc._cli_fallback_target("baremodel") is None


def test_resolvers_case_insensitive():
    assert _resolve_ollama_model("Ollama:foo") == "foo"
    assert _resolve_lmstudio_model("LMStudio:bar") == "bar"
    assert _resolve_ollama_model("plain") == "plain"


def test_lmstudio_base_url_default():
    assert _lmstudio_base_url().endswith("/v1")


async def _empty_fetch(monkeypatch, mod, fn_name):
    import aiohttp

    class _Resp:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Sess:
        closed = False

        def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(mod, "_aiohttp", aiohttp, raising=False)
    return await getattr(mod, fn_name)()


def test_fetch_helpers_exist():
    assert callable(_fetch_ollama_models)
    assert callable(_fetch_lmstudio_models)
