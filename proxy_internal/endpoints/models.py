"""Model listing endpoints (/v1/models, /api/models, /config/*-models)."""

import asyncio
import shutil
import subprocess

import aiohttp
from fastapi import APIRouter

router = APIRouter()


@router.get("/config/ollama-models")
async def get_ollama_models():
    """Run 'ollama list' and return installed model names."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode(errors="replace").splitlines()
        models = []
        for line in lines[1:]:  # skip header row
            name = line.split()[0] if line.split() else ""
            if name:
                models.append(name)
        return {"models": models}
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return {"models": []}


@router.get("/config/zai-models")
async def get_zai_models():
    """Fetch available models from Z.AI API."""
    import proxy

    if not proxy.zai_api_key:
        return {"models": []}
    zai_models = await proxy._fetch_zai_models()
    # Strip the "zai:" prefix since the dashboard JS adds it
    models = [
        m["id"][len("zai:") :] for m in zai_models if m.get("id", "").startswith("zai:")
    ]
    return {"models": models}


@router.get("/config/bytedance-models")
async def get_bytedance_models():
    """Fetch available ByteDance models from the API."""
    import proxy

    if not proxy.bytedance_api_key:
        return {"models": []}
    bytedance_models = await proxy._fetch_bytedance_models()
    models = [
        m["id"][len("bytedance:") :]
        for m in bytedance_models
        if m.get("id", "").startswith("bytedance:")
    ]
    return {"models": models}


@router.get("/config/nvidia-models")
async def get_nvidia_models():
    """Fetch available NVIDIA NIM models from the API."""
    import proxy

    if not proxy.nvidia_api_key:
        return {"models": []}
    nvidia_models = await proxy._fetch_nvidia_models()
    # Strip the "nvidia:" prefix since the dashboard JS adds it
    models = [
        m["id"][len("nvidia:") :]
        for m in nvidia_models
        if m.get("id", "").startswith("nvidia:")
    ]
    return {"models": models}


@router.get("/config/vertex-models")
async def get_vertex_models():
    """List known Vertex AI model aliases for dashboard testing."""
    import proxy
    if not proxy.vertex_api_key:
        return {"models": []}
    from proxy_internal.providers.vertex import _VERTEX_MODEL_ALIASES

    return {"models": list(_VERTEX_MODEL_ALIASES.keys())}


@router.get("/config/kilo-models")
async def get_kilo_models():
    """Fetch live Kilo Gateway model strings for dashboard testing."""
    import proxy

    if not proxy.kilo_api_key:
        return {"models": []}
    kilo_models = await proxy._fetch_kilo_models()
    models = [m["id"] for m in kilo_models if m.get("id", "").startswith("kilo:")]
    return {"models": models}


@router.get("/v1/models")
@router.get("/api/models")
async def list_models():
    """List only models from providers that are currently authenticated, enabled, and active."""
    import proxy
    from proxy_internal.registry import REGISTRY

    data = []

    def _healthy(name: str) -> bool:
        for p in REGISTRY:
            if p.name == name:
                return p.health_fn()
        return False

    def _enabled(name: str) -> bool:
        from proxy_internal import provider_config

        return provider_config.get_provider_setting(name, "enabled", True)

    # Claude
    if _enabled("claude") and _healthy("claude"):
        claude_models = await proxy._fetch_claude_models()
        if claude_models:
            data.extend(claude_models)
        else:
            data.extend(
                [
                    {
                        "id": "claude-opus-4-6",
                        "object": "model",
                        "owned_by": "anthropic",
                    },
                    {
                        "id": "claude-sonnet-4-6",
                        "object": "model",
                        "owned_by": "anthropic",
                    },
                    {
                        "id": "claude-sonnet-4-5-20250929",
                        "object": "model",
                        "owned_by": "anthropic",
                    },
                    {
                        "id": "claude-haiku-4-5-20251001",
                        "object": "model",
                        "owned_by": "anthropic",
                    },
                ]
            )

    # Codex/OpenAI
    if _enabled("codex_cli") and _healthy("codex_cli"):
        codex_models = await proxy._fetch_codex_models()
        if codex_models:
            data.extend(codex_models)
        else:
            data.extend(
                [
                    {"id": "gpt-6-sol", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-6-astra", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-6-luna", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-5.6-sol", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-5.6-terra", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-5.6-luna", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-5.5", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-5.4", "object": "model", "owned_by": "openai"},
                ]
            )

    # Antigravity — only ids live-verified 2026-09-30 (HTTP 200 + real
    # answer content). Removed retired/notice-shaped ids: gemini-3-pro*
    # returns an upstream "no longer available" notice; gemini-3.1-pro
    # (-high) 400s; claude/gpt-oss aliases 404 on this account.
    if _enabled("antigravity") and _healthy("antigravity"):
        data.extend(
            [
                {
                    "id": "antigravity-gemini-3-flash",
                    "object": "model",
                    "owned_by": "google",
                },
                {
                    "id": "antigravity-gemini-3.1-flash",
                    "object": "model",
                    "owned_by": "google",
                },
                {
                    "id": "antigravity-gemini-3.1-flash-lite",
                    "object": "model",
                    "owned_by": "google",
                },
            ]
        )

    # Ollama — only probe when registry says Ollama is healthy
    if _enabled("ollama") and _healthy("ollama"):
        try:
            if proxy.ollama_api_key:
                endpoint = "https://ollama.com/v1/models"
                headers = {"Authorization": f"Bearer {proxy.ollama_api_key}"}
            else:
                endpoint = "http://localhost:11434/v1/models"
                headers = {}
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2, connect=1)
            ) as s:
                async with s.get(endpoint, headers=headers) as ollama_resp:
                    if ollama_resp.status == 200:
                        ollama_data = await ollama_resp.json()
                        for m in ollama_data.get("data", []):
                            model_id = m.get("id", "")
                            if model_id:
                                data.append(
                                    {
                                        "id": f"ollama:{model_id}",
                                        "object": "model",
                                        "owned_by": "ollama",
                                    }
                                )
        except Exception:
            pass

    # LM Studio — only probe when registry says LM Studio is healthy
    if _enabled("lmstudio") and _healthy("lmstudio"):
        try:
            port = proxy.lmstudio_port
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2, connect=1)
            ) as s:
                async with s.get(f"http://localhost:{port}/v1/models") as lmstudio_resp:
                    if lmstudio_resp.status == 200:
                        lmstudio_data = await lmstudio_resp.json()
                        for m in lmstudio_data.get("data", []):
                            model_id = m.get("id", "")
                            if model_id:
                                data.append(
                                    {
                                        "id": f"lmstudio:{model_id}",
                                        "object": "model",
                                        "owned_by": "lmstudio",
                                    }
                                )
        except Exception:
            pass

    # OpenRouter
    if _enabled("openrouter") and _healthy("openrouter"):
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})

    # ByteDance — glm-4.7 superseded by glm-5.1 upstream (provider forwards),
    # so the catalog suggests the live ID only.
    if _enabled("bytedance") and _healthy("bytedance"):
        bytedance_models = await proxy._fetch_bytedance_models()
        if bytedance_models:
            _ALLOWED_BYTEDANCE = frozenset(
                {
                    "dola-seed-2.0-pro",
                    "dola-seed-2.0-lite",
                    "dola-seed-2.0-code",
                    "bytedance-seed-code",
                    "glm-5.1",
                    "kimi-k2.5",
                    "gpt-oss-120b",
                }
            )
            bytedance_models = [
                m
                for m in bytedance_models
                if m.get("id", "").replace("bytedance:", "").lower()
                in {x.lower() for x in _ALLOWED_BYTEDANCE}
            ]
            data.extend(bytedance_models)
        else:
            data.extend(
                [
                    {
                        "id": "bytedance:dola-seed-2.0-pro",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:dola-seed-2.0-lite",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:dola-seed-2.0-code",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:bytedance-seed-code",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:glm-5.1",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:kimi-k2.5",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                    {
                        "id": "bytedance:gpt-oss-120b",
                        "object": "model",
                        "owned_by": "bytedance",
                    },
                ]
            )

    # Kilo Gateway
    if _enabled("kilo") and _healthy("kilo"):
        kilo_models = await proxy._fetch_kilo_models()
        if kilo_models:
            data.extend(kilo_models)
        else:
            data.extend(
                [
                    {
                        "id": "kilo:kilo-auto/balanced",
                        "object": "model",
                        "owned_by": "kilo",
                    },
                    {
                        "id": "kilo:kilo-auto/frontier",
                        "object": "model",
                        "owned_by": "kilo",
                    },
                    {
                        "id": "kilo:kilo-auto/free",
                        "object": "model",
                        "owned_by": "kilo",
                    },
                    {
                        "id": "kilo:kilo-auto/small",
                        "object": "model",
                        "owned_by": "kilo",
                    },
                ]
            )

    # Z.AI
    if _enabled("zai") and _healthy("zai"):
        zai_models = await proxy._fetch_zai_models()
        if zai_models:
            data.extend(zai_models)
        else:
            data.extend(
                [
                    {"id": "zai:glm-4.5-air", "object": "model", "owned_by": "zhipuai"},
                    {"id": "zai:glm-4.5", "object": "model", "owned_by": "zhipuai"},
                    {"id": "zai:glm-4.6", "object": "model", "owned_by": "zhipuai"},
                    {"id": "zai:glm-4.7", "object": "model", "owned_by": "zhipuai"},
                    {"id": "zai:glm-5", "object": "model", "owned_by": "zhipuai"},
                    {"id": "zai:glm-5-turbo", "object": "model", "owned_by": "zhipuai"},
                ]
            )

    # Xiaomi MiMo — v2 generation retired 2026-06-30 (provider aliases them
    # forward); catalog lists live v2.5 + v2.6 IDs only.
    if _enabled("xiaomi") and _healthy("xiaomi"):
        data.extend(
            [
                {"id": "xiaomi:mimo-v2.5-pro", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.5", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.5-asr", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.5-tts", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.5-tts-voiceclone", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.5-tts-voicedesign", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.6-flash", "object": "model", "owned_by": "xiaomi"},
                {"id": "xiaomi:mimo-v2.6-pro", "object": "model", "owned_by": "xiaomi"},
            ]
        )

    # OpenCode Zen — no static ids: the previously advertised free-tier ids
    # (big-pickle, minimax-m2.5-free, nemotron-3-super-free) 401 on this
    # account ("free tier can only be used from within OpenCode") and there
    # is no live Zen catalog endpoint to source replacements from, so the
    # list advertises nothing rather than guessing.

    # OpenCode Go
    if _enabled("opencode_go") and _healthy("opencode_go"):
        from proxy_internal.providers.opencode import _require_go_chat_model

        _GO_CHAT_FALLBACKS = [
            "opencode-go:glm-5.3",
            "opencode-go:kimi-k3",
            "opencode-go:mimo-v2.6-flash",
            "opencode-go:mimo-v2.5-pro",
        ]
        _go_found = 0
        try:
            result = subprocess.run(
                ["opencode", "models", "opencode-go"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("opencode-go/"):
                        api_model = line.split("/", 1)[1]
                        try:
                            _require_go_chat_model(api_model)
                        except Exception:
                            continue
                        data.append(
                            {"id": f"opencode-go:{api_model}", "object": "model", "owned_by": "opencode"}
                        )
                        _go_found += 1
        except Exception:
            pass
        if _go_found == 0:
            data.extend(
                [
                    {
                        "id": mid,
                        "object": "model",
                        "owned_by": "opencode",
                    }
                    for mid in _GO_CHAT_FALLBACKS
                ]
            )

    # Qwen Code
    if _enabled("qwen_cli") and _healthy("qwen_cli"):
        data.extend(
            [
                {"id": "qwen:coder-model", "object": "model", "owned_by": "alibaba"},
            ]
        )

    # Fireworks — short aliases from _FIREWORKS_MODEL_ALIASES.
    if _enabled("fireworks") and _healthy("fireworks"):
        data.extend(
            [
                {
                    "id": f"fireworks:{alias}",
                    "object": "model",
                    "owned_by": "fireworks",
                }
                for alias in [
                    "kimi-k2p5-turbo",
                    "kimi-k3",
                    "deepseek-v4-pro",
                    "deepseek-r1",
                    "minimax-m2p7",
                    "llama-v3p3-70b-instruct",
                ]
            ]
        )

    # NVIDIA NIM
    if _enabled("nvidia") and _healthy("nvidia"):
        nvidia_models = await proxy._fetch_nvidia_models()
        if nvidia_models:
            data.extend(nvidia_models)
        else:
            for alias, full_path in proxy._NVIDIA_MODEL_ALIASES.items():
                data.append(
                    {"id": f"nvidia:{alias}", "object": "model", "owned_by": "nvidia"}
                )
                if alias != full_path:
                    data.append(
                        {
                            "id": f"nvidia:{full_path}",
                            "object": "model",
                            "owned_by": "nvidia",
                        }
                    )

    # Vertex AI
    if _enabled("vertex") and _healthy("vertex"):
        from proxy_internal.providers.vertex import _VERTEX_MODEL_ALIASES

        for alias in _VERTEX_MODEL_ALIASES:
            data.append(
                {"id": f"vertex:{alias}", "object": "model", "owned_by": "google"}
            )

    # Command Code
    if _enabled("command_code") and _healthy("command_code"):
        cmd_models = await proxy._fetch_cmd_models()
        if cmd_models:
            data.extend(cmd_models)
        else:
            data.extend(
                [
                    {"id": f"command_code:{mid}", "object": "model", "owned_by": "command_code"}
                    for mid in [
                        "deepseek-v4-flash", "deepseek-v4-pro",
                        "qwen3.6-max-preview", "qwen3.6-plus", "qwen3.7-max",
                        "kimi-k2.6", "kimi-k2.5",
                        "glm-5.1", "glm-5",
                        "minimax-m2.7", "minimax-m2.5",
                        "step-3.5-flash",
                        "gemini-3.5-flash", "gemini-3.1-flash-lite",
                    ]
                ]
            )

    # Command Code API (key-based) — static documented chat ID, no live request.
    if _enabled("command_api") and _healthy("command_api"):
        data.append(
            {
                "id": "command_api:deepseek/deepseek-v4-flash",
                "object": "model",
                "owned_by": "command_code",
            }
        )

    return {"object": "list", "data": data}


@router.get("/config/command-code-models")
async def get_command_code_models():
    """Fetch available Command Code models from the API."""
    import proxy

    if not proxy.COMMAND_CODE_PATH:
        return {"models": []}
    cmd_models = await proxy._fetch_cmd_models()
    models = [
        m["id"][len("command_code:"):]
        for m in cmd_models
        if m.get("id", "").startswith("command_code:")
    ]
    return {"models": models}


@router.get("/config/vision-models")
async def list_vision_models():
    """Return vision-capable models based on currently available providers.

    Models are filtered by provider availability and returned in a format
    suitable for the dashboard vision test dropdown.
    """
    import proxy

    models = []

    from proxy_internal import provider_config

    def _enabled(name: str) -> bool:
        return provider_config.get_provider_setting(name, "enabled", True)

    # Claude models (vision capable)
    if _enabled("claude") and proxy.auth.is_ready:
        models.extend(
            [
                {
                    "id": "claude-opus-4-6",
                    "name": "Claude Opus 4.6",
                    "provider": "claude",
                },
                {
                    "id": "claude-sonnet-4-6",
                    "name": "Claude Sonnet 4.6",
                    "provider": "claude",
                },
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "name": "Claude Sonnet 4.5",
                    "provider": "claude",
                },
            ]
        )

    # OpenRouter models (vision capable via API key)
    if _enabled("openrouter") and proxy.openrouter_api_key:
        models.extend(
            [
                {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openrouter"},
                {
                    "id": "openai/gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "provider": "openrouter",
                },
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "provider": "openrouter",
                },
                {
                    "id": "anthropic/claude-3-7-sonnet",
                    "name": "Claude 3.7 Sonnet",
                    "provider": "openrouter",
                },
                {
                    "id": "anthropic/claude-3-opus",
                    "name": "Claude 3 Opus",
                    "provider": "openrouter",
                },
                {
                    "id": "google/gemini-3.8-flash",
                    "name": "Gemini 3.8 Flash",
                    "provider": "openrouter",
                },
                {
                    "id": "google/gemini-3.5-flash",
                    "name": "Gemini 3.5 Flash",
                    "provider": "openrouter",
                },
            ]
        )

    # Antigravity models (vision capable Gemini models) — only live-verified ids.
    if proxy.antigravity_auth.is_ready:
        models.extend(
            [
                {
                    "id": "antigravity-gemini-3-flash",
                    "name": "Gemini 3 Flash",
                    "provider": "antigravity",
                },
                {
                    "id": "antigravity-gemini-3.1-flash",
                    "name": "Gemini 3.1 Flash",
                    "provider": "antigravity",
                },
                {
                    "id": "antigravity-gemini-3.1-flash-lite",
                    "name": "Gemini 3.1 Flash Lite",
                    "provider": "antigravity",
                },
            ]
        )

    # Ollama vision models - query CLI for installed models
    ollama_vision_models = []
    try:
        # Check if ollama CLI is available
        ollama_cli = shutil.which("ollama")
        if ollama_cli:
            proc = await asyncio.create_subprocess_exec(
                "ollama",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0:
                output = stdout.decode().strip()
                # Parse table output: NAME ID SIZE MODIFIED
                for line in output.split("\n")[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        model_id = parts[0].strip()
                        # Include known vision-capable models
                        if model_id and any(
                            x in model_id.lower()
                            for x in [
                                "vision",
                                "llava",
                                "moondream",
                                "cogito",
                                "bakllava",
                                "vl",
                            ]
                        ):
                            ollama_vision_models.append(
                                {
                                    "id": f"ollama:{model_id}",
                                    "name": model_id,
                                    "provider": "ollama",
                                }
                            )
    except Exception:
        pass

    # Static fallback list for common vision models
    common_ollama_vision = [
        {"id": "ollama:llava", "name": "llava", "provider": "ollama"},
        {"id": "ollama:llava-phi3", "name": "llava-phi3", "provider": "ollama"},
        {"id": "ollama:llava-llama3", "name": "llava-llama3", "provider": "ollama"},
        {
            "id": "ollama:llama3.2-vision",
            "name": "llama3.2-vision",
            "provider": "ollama",
        },
        {"id": "ollama:bakllava", "name": "bakllava", "provider": "ollama"},
        {"id": "ollama:moondream", "name": "moondream", "provider": "ollama"},
        {
            "id": "ollama:granite3.2-vision",
            "name": "granite3.2-vision",
            "provider": "ollama",
        },
    ]
    # Merge CLI-discovered models with static fallback (avoiding duplicates)
    existing_ids = {m["id"] for m in ollama_vision_models}
    for m in common_ollama_vision:
        if m["id"] not in existing_ids:
            ollama_vision_models.append(m)
    models.extend(ollama_vision_models)

    # Fireworks models (vision capable) - fetched dynamically if possible
    fireworks_vision_models = []
    if proxy.fireworks_api_key:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5, connect=2)
            ) as s:
                headers = {"Authorization": f"Bearer {proxy.fireworks_api_key}"}
                async with s.get(
                    "https://api.fireworks.ai/inference/v1/models", headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("data", []):
                            model_id = m.get("id", "")
                            # Vision models typically have 'vision' or 'vl' in the name
                            if model_id and any(
                                x in model_id.lower()
                                for x in ["vision", "-vl-", "qwen2-vl", "llava"]
                            ):
                                fireworks_vision_models.append(
                                    {
                                        "id": f"fireworks:{model_id}",
                                        "name": m.get("name", model_id),
                                        "provider": "fireworks",
                                    }
                                )
        except Exception:
            pass

        # Static fallback for Fireworks vision models
        # Note: Fireworks uses _resolve_fireworks_model() which strips 'fireworks:' prefix
        # and looks up aliases. Short names like 'kimi-k2p5-turbo' get expanded to full paths.
        if not fireworks_vision_models:
            fireworks_vision_models = [
                # These use the short alias format that gets expanded by _FIREWORKS_MODEL_ALIASES
                {
                    "id": "fireworks:kimi-k2p5-turbo",
                    "name": "Kimi K2.5 Turbo (Vision)",
                    "provider": "fireworks",
                },
            ]
    models.extend(fireworks_vision_models)

    return {"models": models}
