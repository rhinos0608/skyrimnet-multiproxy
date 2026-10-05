"""Current verified model defaults (static checks only — no network).

Each default shipped here was live-verified through the running proxy
(POST /v1/chat/completions, 2xx with non-empty content and a response
`model` field matching the requested provider) before being pinned.
"""

import re

from proxy_internal import model_detect as md
from proxy_internal.registry import REGISTRY as PROVIDERS

EXPECTED_DEFAULTS = {
    "codex_cli": "gpt-6-sol",
    "vertex": "vertex:gemini-3.5-flash",
    "command_code": "command_code:deepseek-v4-flash",
    "openrouter": "anthropic/claude-sonnet-5.5",
    # claude resolves through proxy.DEFAULT_MODEL, which was updated to the
    # live-verified claude-sonnet-5-5 once proxy.py was free to edit.
    "claude": "claude-sonnet-5-5",
    "antigravity": "antigravity-gemini-3-flash",
    "opencode_zen": "opencode:deepseek-v4-flash-free",
    "opencode_go": "opencode-go:glm-5.3",
}

RETIRED_IDS = {
    "gpt-5-codex-mini",
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
    "gpt-5.1",
    "gpt-5.2",
}

DETECTOR_FOR_PROVIDER = None  # detectors are read from each ProviderConfig itself


def _by_name(name):
    for p in PROVIDERS:
        if p.name == name:
            return p
    raise AssertionError(f"provider {name!r} missing from registry")


def test_new_defaults_are_intended_values():
    for name, expected in EXPECTED_DEFAULTS.items():
        assert _by_name(name).default_model == expected, name


def test_codex_default_is_first_verified_catalog_id():
    assert _by_name("codex_cli").default_model == "gpt-6-sol"


def test_every_default_matches_its_own_detector():
    from proxy_internal.registry import detect_provider

    for p in PROVIDERS:
        if p.detector is None:
            # Fallback provider (Claude): default must not be claimed by
            # any other provider, so it genuinely falls through.
            assert detect_provider(p.default_model).name == p.name, p.default_model
        else:
            assert p.detector(p.default_model), (p.name, p.default_model)
        assert detect_provider(p.default_model).name == p.name, (
            p.name,
            p.default_model,
        )


def test_openrouter_default_keeps_bare_vendor_slash_shape():
    default = _by_name("openrouter").default_model
    assert re.match(r"^[^:]+/[^:]+$", default), default
    assert ":" not in default


def test_no_default_uses_retired_id():
    for p in PROVIDERS:
        assert p.default_model not in RETIRED_IDS, (p.name, p.default_model)


def test_retired_codex_ids_absent_from_static_lists():
    from proxy_internal import dashboard as dash_mod
    from proxy_internal.endpoints import dashboard as dash_ep
    from proxy_internal.endpoints import models as models_ep
    import inspect

    for mod in (dash_mod, dash_ep, models_ep):
        src = inspect.getsource(mod)
        for retired in RETIRED_IDS:
            assert f'"{retired}"' not in src and f"'{retired}'" not in src, (
                mod.__name__,
                retired,
            )


def test_codex_detector_accepts_new_catalog_shape():
    for model_id in (
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
    ):
        assert md.is_codex_model(model_id), model_id
    # Bare vendor/model ids must still route to OpenRouter, never Codex.
    assert not md.is_codex_model("openai/gpt-6-luna")
