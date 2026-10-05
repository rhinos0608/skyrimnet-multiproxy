"""Provider alias mapping fixes: Qwen / Z.AI / Xiaomi / ByteDance.

Pins legacy->modern model resolution inside the provider modules only
(shared catalog untouched). Static, no network.
"""

from proxy_internal.providers.qwen import _resolve_qwen_model
from proxy_internal.providers.zai import _resolve_zai_model
from proxy_internal.providers.xiaomi import (
    _resolve_xiaomi_model,
    _xiaomi_base_for,
)
from proxy_internal.providers.bytedance import _resolve_bytedance_model


def test_qwen_legacy_coder_model_resolves_to_portal_default():
    assert _resolve_qwen_model("qwen:coder-model") == "qwen3.5-plus"


def test_qwen_modern_ids_passthrough():
    for mid in ("qwen:qwen3.5-plus", "qwen:qwen3-coder-plus", "qwen:vision-model"):
        assert _resolve_qwen_model(mid) == mid.split(":", 1)[1]


def test_zai_legacy_glm4_resolves_to_glm5():
    assert _resolve_zai_model("zai:glm-4.5-air") == "glm-5.3"
    assert _resolve_zai_model("zai:glm-4.5") == "glm-5.2"
    assert _resolve_zai_model("zai:glm-4.6") == "glm-5-turbo"


def test_zai_modern_ids_passthrough():
    for mid in ("zai:glm-5.3", "zai:glm-5.2", "zai:glm-5", "zai:glm-5-turbo"):
        assert _resolve_zai_model(mid) == mid.split(":", 1)[1]


def test_xiaomi_dead_v2_resolves_to_modern():
    assert _resolve_xiaomi_model("xiaomi:mimo-v2-pro") == "mimo-v2.5-pro"
    assert _resolve_xiaomi_model("xiaomi:mimo-v2-omni") == "mimo-v2.5"
    assert _resolve_xiaomi_model("xiaomi:mimo-v2-flash") == "mimo-v2.6-flash"
    assert _resolve_xiaomi_model("xiaomi:mimo-v2-tts") == "mimo-v2.5-tts"


def test_xiaomi_modern_ids_passthrough_and_routed():
    assert _resolve_xiaomi_model("xiaomi:mimo-v2.5-pro") == "mimo-v2.5-pro"
    assert _resolve_xiaomi_model("xiaomi:mimo-v2.6-pro") == "mimo-v2.6-pro"
    import proxy

    # Resolved next-gen flash must route to the platform endpoint.
    assert _xiaomi_base_for("mimo-v2.6-flash") == proxy.XIAOMI_PLATFORM_URL
    assert _xiaomi_base_for("mimo-v2.5-pro") == proxy.XIAOMI_BASE_URL


def test_bytedance_superseded_glm47_resolves():
    assert _resolve_bytedance_model("bytedance:glm-4.7") == "glm-5.1"


def test_bytedance_dash_variants_resolve_to_live_ids():
    assert _resolve_bytedance_model("bytedance:seed-2-0-pro") == "dola-seed-2.0-pro"
    assert _resolve_bytedance_model("bytedance:seed-2-0-lite") == "dola-seed-2.0-lite"
    assert _resolve_bytedance_model("bytedance:seed-2-0-code") == "dola-seed-2.0-code"


def test_bytedance_live_ids_passthrough():
    for mid in (
        "bytedance:dola-seed-2.0-pro",
        "bytedance:bytedance-seed-code",
        "bytedance:glm-5.1",
        "bytedance:kimi-k2.5",
    ):
        assert _resolve_bytedance_model(mid) == mid.split(":", 1)[1]
