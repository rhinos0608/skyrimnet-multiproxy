"""Focused mapping tests for Fireworks / Kilo / NVIDIA provider aliases."""

from proxy_internal.providers.fireworks import _resolve_fireworks_model
from proxy_internal.providers.kilo import _resolve_kilo_model
from proxy_internal.providers.nvidia import (
    _get_thinking_kwargs,
    _resolve_nvidia_model,
)


def test_fireworks_new_aliases():
    assert _resolve_fireworks_model("fireworks:kimi-k3") == "accounts/fireworks/models/kimi-k3"
    assert _resolve_fireworks_model("fireworks:deepseek-v4-pro") == "accounts/fireworks/models/deepseek-v4-pro"
    assert _resolve_fireworks_model("fireworks:deepseek-r1") == "accounts/fireworks/models/deepseek-r1"
    assert _resolve_fireworks_model("fireworks:minimax-m2p7") == "accounts/fireworks/models/minimax-m2p7"
    assert _resolve_fireworks_model("fireworks:llama-v3p3-70b-instruct") == (
        "accounts/fireworks/models/llama-v3p3-70b-instruct"
    )


def test_fireworks_legacy_alias_kept():
    assert _resolve_fireworks_model("fireworks:kimi-k2p5-turbo") == (
        "accounts/fireworks/routers/kimi-k2p5-turbo"
    )


def test_kilo_passthrough():
    # Kilo gateway is dynamic (/models); resolver only strips prefix.
    assert _resolve_kilo_model("kilo:kilo-auto/efficient") == "kilo-auto/efficient"
    assert _resolve_kilo_model("kilo:kilo-auto/balanced") == "kilo-auto/balanced"
    assert _resolve_kilo_model("kilo:anthropic/claude-opus-5.5") == "anthropic/claude-opus-5.5"


def test_nvidia_new_aliases():
    assert _resolve_nvidia_model("nvidia:kimi-k3") == "moonshotai/kimi-k3"
    assert _resolve_nvidia_model("nvidia:deepseek-v4.1-flash") == "deepseek-ai/deepseek-v4.1-flash"
    assert _resolve_nvidia_model("nvidia:glm-5.3") == "z-ai/glm-5.3"
    assert _resolve_nvidia_model("nvidia:mistral-small-4") == "mistralai/mistral-small-4-119b-2603"
    assert _resolve_nvidia_model("nvidia:nemotron-3.5-lightning") == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert _resolve_nvidia_model("nvidia:nemotron-3-ultra") == "nvidia/nemotron-3-ultra-550b-a55b"
    assert _resolve_nvidia_model("nvidia:step-3.7-flash") == "stepfun-ai/step-3.7-flash"


def test_nvidia_legacy_aliases_kept():
    assert _resolve_nvidia_model("nvidia:mistral-large-3") == "mistralai/mistral-large-3-675b-instruct-2512"
    assert _resolve_nvidia_model("nvidia:llama-4-maverick") == "meta/llama-4-maverick-17b-128e-instruct"
    assert _resolve_nvidia_model("nvidia:qwen3-coder-480b") == "qwen/qwen3-coder-480b-a35b-instruct"


def test_nvidia_thinking_kwargs_new_families():
    assert _get_thinking_kwargs("moonshotai/kimi-k3", "low") == {"thinking": True}
    assert _get_thinking_kwargs("z-ai/glm-5.3", "low") == {"enable_thinking": True, "clear_thinking": False}
