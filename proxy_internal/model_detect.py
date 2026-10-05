"""Model-name prefix/pattern detection and alias normalisation."""


def parse_model_list(model_field: str) -> list[str]:
    """Parse comma-separated model list from request, trimming whitespace."""
    return [m.strip() for m in model_field.split(",") if m.strip()]


def is_ollama_model(model: str) -> bool:
    """Ollama models use 'ollama:model' or 'ollama:model:tag' prefix."""
    return model.lower().startswith("ollama:")


def is_lmstudio_model(model: str) -> bool:
    """LM Studio models use 'lmstudio:model' prefix."""
    return model.lower().startswith("lmstudio:")


def is_bytedance_model(model: str) -> bool:
    """ByteDance models use 'bytedance:model' prefix."""
    return model.lower().startswith("bytedance:")


def is_openrouter_model(model: str) -> bool:
    """OpenRouter models use 'provider/model' format (contain '/').
    Excludes models with provider prefixes (colon before first slash),
    but allows OpenRouter suffix tags like ':free', ':beta' after the slash."""
    # A colon in the part before the first slash indicates a provider prefix
    # (e.g. ollama:model, command_code:foo/bar).  If present it is NOT OpenRouter.
    # OpenRouter IDs are vendor/model with optional :suffix after the slash
    # (e.g. meta-llama/llama-3-8b:free).
    if ":" in model.split("/", 1)[0]:
        return False
    return "/" in model


def is_codex_model(model: str) -> bool:
    """Codex/OpenAI models use gpt-*/gpt-5.*-codex* naming or codex-* naming."""
    model_lower = model.lower()
    return (
        model_lower == "gpt-5" or
        model_lower.startswith("gpt-") or
        model_lower.startswith("codex-")
    )


def is_antigravity_model(model: str) -> bool:
    """Antigravity models use antigravity-* naming."""
    model_lower = model.lower()
    return (
        model_lower.startswith("antigravity-") or
        model_lower.startswith("gemini-3") or
        model_lower.startswith("gemini-2.5") or
        model_lower.startswith("gpt-oss-")
    )


def is_zai_model(model: str) -> bool:
    """Z.AI models use 'zai:model' prefix."""
    return model.lower().startswith("zai:")


def is_xiaomi_model(model: str) -> bool:
    """Xiaomi models use 'xiaomi:model' prefix."""
    return model.lower().startswith("xiaomi:")


def is_opencode_model(model: str) -> bool:
    """OpenCode models use 'opencode:model' or 'opencode-go:model' prefix."""
    m = model.lower()
    return m.startswith("opencode:") or m.startswith("opencode-go:")


def is_opencode_zen_model(model: str) -> bool:
    """OpenCode Zen models use 'opencode:' prefix (NOT opencode-go:)."""
    m = model.lower()
    return m.startswith("opencode:") and not m.startswith("opencode-go:")


def is_opencode_go_model(model: str) -> bool:
    """OpenCode Go models use 'opencode-go:' prefix."""
    return model.lower().startswith("opencode-go:")


def is_qwen_model(model: str) -> bool:
    """Qwen Code models use 'qwen:model' prefix."""
    return model.lower().startswith("qwen:")


def is_fireworks_model(model: str) -> bool:
    """Fireworks models use 'fireworks:model' prefix."""
    return model.lower().startswith("fireworks:")


def is_nvidia_model(model: str) -> bool:
    """NVIDIA NIM models use 'nvidia:model' prefix."""
    return model.lower().startswith("nvidia:")


def is_kilo_model(model: str) -> bool:
    """Kilo Gateway models use 'kilo:model' prefix."""
    return model.lower().startswith("kilo:")


def is_command_code_model(model: str) -> bool:
    """Command Code models use 'command_code:model' prefix."""
    return model.lower().startswith("command_code:")


def is_command_code_api_model(model: str) -> bool:
    """Command Code API models use 'command_api:vendor/model' prefix (case insensitive)."""
    return isinstance(model, str) and model.lower().startswith("command_api:")


def is_vertex_model(model: str) -> bool:
    """Vertex AI models use 'vertex:model' prefix."""
    return model.lower().startswith("vertex:")


# Canonical model name aliases — dot-version notation → hyphen notation used by the API
_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
}


def normalize_model_name(model: str) -> str:
    """Resolve known model aliases to their canonical API identifiers."""
    return _MODEL_ALIASES.get(model, model)
