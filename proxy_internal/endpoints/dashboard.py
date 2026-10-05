"""Dashboard route handler."""
import html
import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from proxy_internal.dashboard import DashboardContext, render_dashboard

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    import proxy  # lazy — read module state at request time

    auth = proxy.auth
    codex_auth = proxy.codex_auth
    qwen_auth = proxy.qwen_auth
    antigravity_auth = proxy.antigravity_auth

    from proxy_internal import provider_config

    def _disabled(name: str) -> bool:
        return not provider_config.get_provider_setting(name, "enabled", True)

    _any_ready = (
        auth.is_ready
        or codex_auth.is_ready
        or antigravity_auth.is_ready
        or proxy.openrouter_api_key
        or proxy.kilo_api_key
        or proxy.ollama_api_key
        or proxy.zai_api_key
        or proxy.xiaomi_api_key
        or proxy.opencode_api_key
        or proxy.opencode_go_api_key
        or qwen_auth.is_ready
        or proxy.fireworks_api_key
        or proxy.nvidia_api_key
        or proxy.bytedance_api_key
        or proxy.COMMAND_CODE_PATH
    )
    status = "Ready" if _any_ready else "Warming up..."
    status_color = "#4ade80" if _any_ready else "#facc15"

    if _disabled("openrouter"):
        or_status = "Disabled"
        or_color = "#94a3b8"
    else:
        or_status = "Configured (saved)" if proxy.openrouter_api_key else "Not set"
        or_color = "#4ade80" if proxy.openrouter_api_key else "#64748b"

    if _disabled("kilo"):
        kilo_status = "Disabled"
        kilo_color = "#94a3b8"
    else:
        kilo_status = "Configured (saved)" if proxy.kilo_api_key else "Not set"
        kilo_color = "#4ade80" if proxy.kilo_api_key else "#64748b"

    if _disabled("ollama"):
        ollama_status = "Disabled"
        ollama_color = "#94a3b8"
    else:
        ollama_status = "Cloud (key configured)" if proxy.ollama_api_key else "Local (localhost:11434)"
        ollama_color = "#4ade80" if proxy.ollama_api_key else "#64748b"

    if _disabled("lmstudio"):
        lmstudio_status = "Disabled"
        lmstudio_color = "#94a3b8"
    else:
        from proxy_internal.registry import _is_lmstudio_healthy
        lmstudio_healthy = _is_lmstudio_healthy()
        lmstudio_status = f"Local (localhost:{proxy.lmstudio_port})" if lmstudio_healthy else "Not running"
        lmstudio_color = "#4ade80" if lmstudio_healthy else "#64748b"

    if _disabled("zai"):
        zai_status = "Disabled"
        zai_color = "#94a3b8"
    else:
        zai_status = "Configured (saved)" if proxy.zai_api_key else "Not set"
        zai_color = "#4ade80" if proxy.zai_api_key else "#64748b"

    if _disabled("xiaomi"):
        xiaomi_status = "Disabled"
        xiaomi_color = "#94a3b8"
    else:
        xiaomi_status = "Configured (saved)" if proxy.xiaomi_api_key else "Not set"
        xiaomi_color = "#4ade80" if proxy.xiaomi_api_key else "#64748b"

    if _disabled("opencode_zen"):
        opencode_status = "Disabled"
        opencode_color = "#94a3b8"
    else:
        _oc_ready = proxy.opencode_api_key or proxy.opencode_go_api_key
        opencode_status = (
            f"{'Zen' if proxy.opencode_api_key else ''}"
            f"{' + ' if proxy.opencode_api_key and proxy.opencode_go_api_key else ''}"
            f"{'Go' if proxy.opencode_go_api_key else ''}"
            if _oc_ready else "Not set"
        )
        opencode_color = "#4ade80" if _oc_ready else "#64748b"

    if _disabled("qwen_cli"):
        qwen_status = "Disabled"
        qwen_color = "#94a3b8"
    else:
        qwen_status = (
            "Ready" if qwen_auth.is_ready
            else ("Credentials found" if os.path.exists(proxy.QWEN_CREDS_FILE) else "Not authenticated")
        )
        qwen_color = (
            "#4ade80" if qwen_auth.is_ready
            else ("#facc15" if os.path.exists(proxy.QWEN_CREDS_FILE) else "#64748b")
        )

    if _disabled("fireworks"):
        fireworks_status = "Disabled"
        fireworks_color = "#94a3b8"
    else:
        fireworks_status = "Configured (saved)" if proxy.fireworks_api_key else "Not set"
        fireworks_color = "#4ade80" if proxy.fireworks_api_key else "#64748b"

    if _disabled("nvidia"):
        nvidia_status = "Disabled"
        nvidia_color = "#94a3b8"
    else:
        nvidia_status = "Configured (saved)" if proxy.nvidia_api_key else "Not set"
        nvidia_color = "#4ade80" if proxy.nvidia_api_key else "#64748b"

    if _disabled("bytedance"):
        bytedance_status = "Disabled"
        bytedance_color = "#94a3b8"
    else:
        bytedance_status = "Configured (saved)" if proxy.bytedance_api_key else "Not set"
        bytedance_color = "#4ade80" if proxy.bytedance_api_key else "#64748b"

    if _disabled("vertex"):
        vertex_status = "Disabled"
        vertex_color = "#94a3b8"
    else:
        vertex_status = "Configured (saved)" if proxy.vertex_api_key else "Not set"
        vertex_color = "#4ade80" if proxy.vertex_api_key else "#64748b"

    if _disabled("command_code"):
        command_code_status = "Disabled"
        command_code_color = "#94a3b8"
    else:
        command_code_status = "Installed" if proxy.COMMAND_CODE_PATH else "Not installed"
        command_code_color = "#4ade80" if proxy.COMMAND_CODE_PATH else "#64748b"

    if _disabled("codex_cli"):
        codex_status = "Disabled"
        codex_color = "#94a3b8"
    else:
        codex_status = (
            "Ready" if codex_auth.is_ready
            else ("Not authenticated" if proxy.CODEX_PATH else "Not installed")
        )
        codex_color = (
            "#4ade80" if codex_auth.is_ready
            else ("#facc15" if proxy.CODEX_PATH else "#64748b")
        )

    if _disabled("claude"):
        claude_status = "Disabled"
        claude_color = "#94a3b8"
    else:
        claude_status = (
            "Ready" if auth.is_ready
            else ("Not authenticated" if proxy.CLAUDE_PATH else "Not installed")
        )
        claude_color = (
            "#4ade80" if auth.is_ready
            else ("#facc15" if proxy.CLAUDE_PATH else "#64748b")
        )

    if _disabled("antigravity"):
        ag_status = "Disabled"
        ag_color = "#94a3b8"
    else:
        ag_status = "Ready" if antigravity_auth.is_ready else "Not authenticated"
        ag_color = "#4ade80" if antigravity_auth.is_ready else "#facc15"
    ag_accounts = antigravity_auth.get_all_accounts_info()
    ag_account_count = len(ag_accounts)
    # Build account list HTML
    if ag_accounts:
        ag_accounts_html = "<div style='margin-top:8px;font-size:0.85rem'>"
        for acc in ag_accounts:
            safe_email = html.escape(acc['email'])
            # For JS string context, also escape backslashes and quotes
            js_safe_email = safe_email.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
            status_icon = "✓" if acc["is_ready"] else "⚠"
            status_color_acc = "#4ade80" if acc["is_ready"] else "#facc15"
            error_badge = (
                f" <span style='color:#f87171;font-size:0.75rem'>({acc['error_count']} errors)</span>"
                if acc.get("error_count", 0) > 0 else ""
            )
            remove_link = (
                f"<a href='#' onclick='removeAccount(\"{js_safe_email}\");return false' "
                f"style='color:#f87171;margin-left:8px;font-size:0.75rem'>[remove]</a>"
            )
            ag_accounts_html += (
                f"<div style='margin:4px 0'><span style='color:{status_color_acc}'>{status_icon}</span> "
                f"{safe_email}{error_badge}{remove_link}</div>"
            )
        ag_accounts_html += "</div>"
    else:
        ag_accounts_html = ""

    # Claude models
    claude_models = [
        ("claude-opus-4-6", "Opus 4.6", "Most capable"),
        ("claude-sonnet-4-6", "Sonnet 4.6", "Default"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "Previous"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest"),
    ]
    claude_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in claude_models
    )

    # Codex models
    codex_models = []
    if codex_auth.is_ready:
        try:
            codex_models_raw = proxy._fetch_codex_models()
            # If it's a coroutine, we need to await it. But we're in async function!
            codex_models_raw = await codex_models_raw
            for m in codex_models_raw:
                mid = m["id"]
                name = m.get("name") or mid
                desc = m.get("owned_by") or "Available"
                codex_models.append((mid, name, desc))
        except Exception:
            pass

    if not codex_models:
        codex_models = [
            ("gpt-6-sol", "GPT-6 Sol", "Default"),
            ("gpt-6-astra", "GPT-6 Astra", "Code-optimized"),
            ("gpt-6-luna", "GPT-6 Luna", "Fast"),
            ("gpt-5.6-sol", "GPT-5.6 Sol", "Code-optimized"),
            ("gpt-5.6-terra", "GPT-5.6 Terra", "General purpose"),
            ("gpt-5.6-luna", "GPT-5.6 Luna", "Fast"),
            ("gpt-5.5", "GPT-5.5", "General purpose"),
            ("gpt-5.4", "GPT-5.4", "Flagship frontier"),
        ]

    codex_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in codex_models
    )

    # Antigravity models — only ids live-verified 2026-09-30 (matches /v1/models).
    antigravity_models = [
        ("antigravity-gemini-3-flash", "Gemini 3 Flash", "Fast"),
        ("antigravity-gemini-3.1-flash", "Gemini 3.1 Flash", "Latest"),
        ("antigravity-gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", "Fastest"),
    ]
    antigravity_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in antigravity_models
    )

    # Card class hints
    claude_card_class = "provider-ready" if auth.is_ready else "provider-notready"
    claude_status_class = (
        "status-ready" if auth.is_ready
        else ("status-notready" if proxy.CLAUDE_PATH else "status-offline")
    )
    codex_card_class = "provider-ready" if codex_auth.is_ready else "provider-notready"
    codex_status_class = (
        "status-ready" if codex_auth.is_ready
        else ("status-notready" if proxy.CODEX_PATH else "status-offline")
    )
    ag_card_class = "provider-ready" if antigravity_auth.is_ready else "provider-notready"
    ag_status_class = "status-ready" if antigravity_auth.is_ready else "status-notready"
    qwen_card_class = "provider-ready" if qwen_auth.is_ready else "provider-notready"
    qwen_status_class = (
        "status-ready" if qwen_auth.is_ready
        else ("status-notready" if os.path.exists(proxy.QWEN_CREDS_FILE) else "status-offline")
    )

    qwen_hint = (
        '<div style="color:#64748b;font-size:0.8rem;margin-bottom:8px">'
        'Run <code style="color:#67e8f9">qwen code</code> and log in, then restart proxy.'
        '</div>'
        if not qwen_auth.is_ready else ''
    )

    opencode_go_status_color = "#4ade80" if proxy.opencode_go_api_key else "#64748b"
    opencode_go_status_text = "Configured" if proxy.opencode_go_api_key else "Not set"

    timeout_routing_checked = "checked" if proxy.timeout_routing_enabled else ""
    timeout_routing_color = "#4ade80" if proxy.timeout_routing_enabled else "#64748b"
    timeout_routing_label = "Active" if proxy.timeout_routing_enabled else "Disabled"

    reasoning_override_enabled = bool(proxy.reasoning_override_enabled)
    reasoning_override_level = proxy.reasoning_override_level
    reasoning_override_checked = "checked" if reasoning_override_enabled else ""
    reasoning_override_color = "#4ade80" if reasoning_override_enabled else "#64748b"
    reasoning_override_label = (
        f"Active ({reasoning_override_level})" if reasoning_override_enabled else "Disabled"
    )
    reasoning_override_badge = (
        f'<span class="status" style="background:#a78bfa20;color:#a78bfa;'
        f'border:1px solid #a78bfa40;margin-left:8px">🧠 Reasoning: {reasoning_override_level}</span>'
        if reasoning_override_enabled else ""
    )

    ctx = DashboardContext(
        status=status,
        status_color=status_color,
        claude_status=claude_status,
        claude_color=claude_color,
        codex_status=codex_status,
        codex_color=codex_color,
        qwen_status=qwen_status,
        qwen_color=qwen_color,
        ollama_status=ollama_status,
        ollama_color=ollama_color,
        lmstudio_status=lmstudio_status,
        lmstudio_color=lmstudio_color,
        zai_status=zai_status,
        zai_color=zai_color,
        xiaomi_status=xiaomi_status,
        xiaomi_color=xiaomi_color,
        opencode_status=opencode_status,
        opencode_color=opencode_color,
        fireworks_status=fireworks_status,
        fireworks_color=fireworks_color,
        nvidia_status=nvidia_status,
        nvidia_color=nvidia_color,
        bytedance_status=bytedance_status,
        bytedance_color=bytedance_color,
        vertex_status=vertex_status,
        vertex_color=vertex_color,
        command_code_status=command_code_status,
        command_code_color=command_code_color,
        or_status=or_status,
        or_color=or_color,
        kilo_status=kilo_status,
        kilo_color=kilo_color,
        ag_status=ag_status,
        ag_color=ag_color,
        ag_account_count=ag_account_count,
        claude_rows=claude_rows,
        codex_rows=codex_rows,
        antigravity_rows=antigravity_rows,
        ag_accounts_html=ag_accounts_html,
        claude_card_class=claude_card_class,
        claude_status_class=claude_status_class,
        codex_card_class=codex_card_class,
        codex_status_class=codex_status_class,
        ag_card_class=ag_card_class,
        ag_status_class=ag_status_class,
        qwen_card_class=qwen_card_class,
        qwen_status_class=qwen_status_class,
        qwen_hint=qwen_hint,
        opencode_go_status_color=opencode_go_status_color,
        opencode_go_status_text=opencode_go_status_text,
        timeout_routing_enabled=proxy.timeout_routing_enabled,
        timeout_cutoff_seconds=proxy.timeout_cutoff_seconds,
        max_total_seconds=proxy.max_total_seconds,
        max_retries=proxy.max_retries,
        timeout_routing_checked=timeout_routing_checked,
        timeout_routing_color=timeout_routing_color,
        timeout_routing_label=timeout_routing_label,
        reasoning_override_enabled=reasoning_override_enabled,
        reasoning_override_level=reasoning_override_level,
        reasoning_override_checked=reasoning_override_checked,
        reasoning_override_color=reasoning_override_color,
        reasoning_override_label=reasoning_override_label,
        reasoning_override_badge=reasoning_override_badge,
        reasoning_rewrite_enabled=bool(proxy.reasoning_rewrite_enabled),
        reasoning_rewrite_checked="checked" if proxy.reasoning_rewrite_enabled else "",
        reasoning_rewrite_color="#4ade80" if proxy.reasoning_rewrite_enabled else "#64748b",
        reasoning_rewrite_label="Active" if proxy.reasoning_rewrite_enabled else "Disabled",
    )
    response = HTMLResponse(content=render_dashboard(ctx))
    response.set_cookie(
        key=proxy.ADMIN_CSRF_COOKIE,
        value=proxy.ADMIN_CSRF_TOKEN,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response
