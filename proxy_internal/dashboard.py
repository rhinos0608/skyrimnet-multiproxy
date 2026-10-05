"""Dashboard HTML template rendering."""

from dataclasses import dataclass


@dataclass
class DashboardContext:
    """Pre-computed state for dashboard template rendering.

    All fields are strings / primitives that can be directly substituted
    into the HTML f-string template — no further conditionals required at
    render time.
    """

    # Overall status badge
    status: str
    status_color: str

    # Per-provider status + color pairs (used in inline style attributes)
    claude_status: str
    claude_color: str
    codex_status: str
    codex_color: str
    qwen_status: str
    qwen_color: str
    ollama_status: str
    ollama_color: str
    lmstudio_status: str
    lmstudio_color: str
    zai_status: str
    zai_color: str
    xiaomi_status: str
    xiaomi_color: str
    opencode_status: str
    opencode_color: str
    fireworks_status: str
    fireworks_color: str
    nvidia_status: str
    nvidia_color: str
    bytedance_status: str
    bytedance_color: str
    vertex_status: str
    vertex_color: str
    command_code_status: str
    command_code_color: str
    or_status: str
    or_color: str
    kilo_status: str
    kilo_color: str
    ag_status: str
    ag_color: str
    ag_account_count: int

    # Pre-rendered HTML blocks
    claude_rows: str
    codex_rows: str
    antigravity_rows: str
    ag_accounts_html: str

    # Provider card class hints (so template avoids inline conditionals)
    claude_card_class: str
    claude_status_class: str
    codex_card_class: str
    codex_status_class: str
    ag_card_class: str
    ag_status_class: str
    qwen_card_class: str
    qwen_status_class: str

    # Inline hint snippets
    qwen_hint: str

    # OpenCode Go status (separate save-panel label)
    opencode_go_status_color: str
    opencode_go_status_text: str

    # Runtime config values (raw)
    timeout_routing_enabled: bool
    timeout_cutoff_seconds: float
    max_total_seconds: float
    max_retries: int
    timeout_routing_checked: str
    timeout_routing_color: str
    timeout_routing_label: str

    # Reasoning override (forces thinking on regardless of caller input — used
    # to rescue CCS-routed Claude Code sessions that land here with reasoning
    # silently disabled).
    reasoning_override_enabled: bool
    reasoning_override_level: str
    reasoning_override_checked: str
    reasoning_override_color: str
    reasoning_override_label: str
    reasoning_override_badge: str

    # Reasoning rewrite (secondary Fireworks call for reasoning-only responses)
    reasoning_rewrite_enabled: bool
    reasoning_rewrite_checked: str
    reasoning_rewrite_color: str
    reasoning_rewrite_label: str


def render_dashboard(ctx: DashboardContext) -> str:
    """Render the dashboard HTML from pre-computed state."""
    return f"""<!DOCTYPE html>
<html><head><title>Claude SkyrimNet Proxy</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:800px; margin:40px auto; padding:0 20px }}
  h1 {{ color:#f8fafc; font-size:1.5rem; margin-bottom:4px }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:30px }}
  .status {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:0.85rem; font-weight:600;
             background:{ctx.status_color}20; color:{ctx.status_color}; border:1px solid {ctx.status_color}40 }}
  .card {{ background:#1e293b; border-radius:8px; padding:20px; margin:16px 0; border:1px solid #334155 }}
  .provider-card {{ background:#1e293b; border-radius:8px; padding:16px; border:1px solid #334155 }}
  .provider-ready {{ border-color:#4ade8040 }}
  .provider-notready {{ border-color:#facc1540 }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ text-align:left; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; padding:8px 12px; border-bottom:1px solid #334155 }}
  td {{ padding:6px 12px; border-bottom:1px solid #1e293b }}
  .label {{ color:#94a3b8; font-size:0.85rem }}
  .value {{ color:#f1f5f9; font-family:monospace; font-size:0.85rem }}
  .endpoint {{ background:#0f172a; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:0.85rem; color:#67e8f9; margin:8px 0; border:1px solid #334155 }}
  textarea {{ width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:10px; font-family:monospace; font-size:0.85rem; resize:vertical; box-sizing:border-box }}
  button {{ background:#3b82f6; color:white; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin-top:8px }}
  button:hover {{ background:#2563eb }}
  button:disabled {{ background:#475569; cursor:wait }}
  #response {{ margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap }}
  .timing {{ color:#4ade80; font-size:0.8rem; margin-top:6px }}
  select {{ background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem }}
  .provider-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0 }}
  .provider-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px }}
  .provider-status {{ padding:4px 10px; border-radius:12px; font-size:0.75rem; font-weight:600 }}
  .status-ready {{ background:#4ade8020; color:#4ade80 }}
  .status-notready {{ background:#facc1520; color:#facc15 }}
  .status-offline {{ background:#64748b20; color:#64748b }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style></head>
<body>
  <h1>Claude SkyrimNet Proxy</h1>
  <div class="subtitle">OpenAI-compatible proxy using Claude Max, Codex CLI, or Kilo Gateway</div>
  <span class="status">{ctx.status}</span>
  {ctx.reasoning_override_badge}

  <div class="card" style="border-color:#f87171;background:#450a0a20">
    <h3 style="margin:0 0 8px;font-size:1rem;color:#f87171">⚠️ Terms of Service Warning</h3>
    <p style="margin:0;color:#fca5a5;font-size:0.85rem">
      The Codex provider intercepts OAuth tokens from the official Codex CLI. The Antigravity provider uses
      Google OAuth to access the Antigravity IDE API. These approaches exist in a
      <strong>gray area of Terms of Service</strong>. Use at your own risk.
    </p>
  </div>

  <div class="card">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div><span class="label">Endpoint</span><div class="endpoint">https://macbook-pro.taila2c1ae.ts.net:8539/v1/chat/completions</div></div>
      <div><span class="label">API Key</span><div class="endpoint">not required</div></div>
    </div>
  </div>

  <!-- Provider Slots -->
  <div class="provider-grid">
    <!-- Claude Provider -->
    <div class="provider-card {ctx.claude_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Claude (Anthropic)</h3>
        <span class="provider-status {ctx.claude_status_class}">{ctx.claude_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">MITM Interceptor</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.claude_rows}</tbody>
      </table>
    </div>

    <!-- Codex Provider -->
    <div class="provider-card {ctx.codex_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 Codex (OpenAI)</h3>
        <span class="provider-status {ctx.codex_status_class}">{ctx.codex_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">Isolated HOME</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.codex_rows}</tbody>
      </table>
    </div>

    <!-- Antigravity Provider -->
    <div class="provider-card {ctx.ag_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔵 Antigravity (Google)</h3>
        <span class="provider-status {ctx.ag_status_class}">{ctx.ag_status}</span>
      </div>
      <div style="margin-bottom:8px">
        <span class="label">Auth Method:</span> <span class="value">Google OAuth</span>
        <span style="margin-left:12px" class="label">Accounts:</span> <span class="value">{ctx.ag_account_count}</span>
      </div>
      {ctx.ag_accounts_html}
      <div style="margin:8px 0">
        <a href="/config/antigravity-login" style="color:#67e8f9;font-size:0.85rem" target="_blank">+ Add Google Account →</a>
      </div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.antigravity_rows}</tbody>
      </table>
    </div>

    <!-- Ollama Provider -->
    <div class="provider-card" style="border-color:{ctx.ollama_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🦙 Ollama</h3>
        <span class="provider-status" style="background:{ctx.ollama_color}20;color:{ctx.ollama_color}">{ctx.ollama_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">ollama:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pull models with <code style="color:#67e8f9">ollama pull model-name</code><br>
        Set API key below for Ollama Cloud.
      </div>
    </div>

    <!-- LM Studio Provider -->
    <div class="provider-card" style="border-color:{ctx.lmstudio_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🖥️ LM Studio</h3>
        <span class="provider-status" style="background:{ctx.lmstudio_color}20;color:{ctx.lmstudio_color}">{ctx.lmstudio_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">lmstudio:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Load a model in LM Studio and start the local server.<br>
        e.g. <code style="color:#67e8f9">lmstudio:my-model-name</code>
      </div>
    </div>

    <!-- Z.AI Provider -->
    <div class="provider-card" style="border-color:{ctx.zai_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">⚡ Z.AI</h3>
        <span class="provider-status" style="background:{ctx.zai_color}20;color:{ctx.zai_color}">{ctx.zai_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">zai:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.z.ai/api/coding/paas/v4</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        GLM models via Z.AI coding plan.<br>
        e.g. <code style="color:#67e8f9">zai:glm-4.7</code>, <code style="color:#67e8f9">zai:glm-5</code>
      </div>
    </div>

    <!-- Xiaomi MiMo Provider -->
    <div class="provider-card" style="border-color:{ctx.xiaomi_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Xiaomi MiMo</h3>
        <span class="provider-status" style="background:{ctx.xiaomi_color}20;color:{ctx.xiaomi_color}">{ctx.xiaomi_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">xiaomi:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">token-plan-sgp.xiaomimimo.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pro/Omni → token-plan-sgp endpoint.<br>
        Flash/TTS/ASR → api.xiaomimimo.com (platform models).<br>
        e.g. <code style="color:#67e8f9">xiaomi:mimo-v2.5-pro</code>, <code style="color:#67e8f9">xiaomi:mimo-v2.5</code>
      </div>
    </div>

    <!-- OpenCode Provider -->
    <div class="provider-card" style="border-color:{ctx.opencode_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🌐 OpenCode</h3>
        <span class="provider-status" style="background:{ctx.opencode_color}20;color:{ctx.opencode_color}">{ctx.opencode_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Zen:</span> <span class="value">opencode:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Go:</span> <span class="value">opencode-go:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Zen: 40+ curated models (Claude, Gemini, GPT, GLM, Qwen).<br>
        Go: Kimi, GLM, MiMo, Minimax.<br>
        e.g. <code style="color:#67e8f9">opencode-go:kimi-k2.5</code>, <code style="color:#67e8f9">opencode:glm-5</code>
      </div>
    </div>

    <!-- Qwen Code Provider -->
    <div class="provider-card {ctx.qwen_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟣 Qwen Code</h3>
        <span class="provider-status {ctx.qwen_status_class}">{ctx.qwen_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">~/.qwen/oauth_creds.json</span></div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">qwen:model-name</span></div>
      {ctx.qwen_hint}
      <div style="color:#64748b;font-size:0.8rem">
        Free tier: 1,000 req/day via Qwen OAuth.<br>
        e.g. <code style="color:#67e8f9">qwen:coder-model</code>
      </div>
    </div>

    <!-- Fireworks Provider -->
    <div class="provider-card" style="border-color:{ctx.fireworks_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔥 Fireworks</h3>
        <span class="provider-status" style="background:{ctx.fireworks_color}20;color:{ctx.fireworks_color}">{ctx.fireworks_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">fireworks:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.fireworks.ai/inference/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        OpenAI-compatible API for hosted models.<br>
        e.g. <code style="color:#67e8f9">fireworks:accounts/fireworks/routers/kimi-k2p5-turbo</code>
      </div>
    </div>

    <!-- NVIDIA NIM Provider -->
    <div class="provider-card" style="border-color:{ctx.nvidia_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 NVIDIA NIM</h3>
        <span class="provider-status" style="background:{ctx.nvidia_color}20;color:{ctx.nvidia_color}">{ctx.nvidia_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">nvidia:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">integrate.api.nvidia.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Free-tier OpenAI-compatible API for NVIDIA-hosted models.<br>
        e.g. <code style="color:#67e8f9">nvidia:deepseek-r1</code>, <code style="color:#67e8f9">nvidia:llama-3.3-70b</code>
      </div>
    </div>

    <!-- ByteDance Provider -->
    <div class="provider-card" style="border-color:{ctx.bytedance_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔵 ByteDance</h3>
        <span class="provider-status" style="background:{ctx.bytedance_color}20;color:{ctx.bytedance_color}">{ctx.bytedance_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">bytedance:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">ark.ap-southeast.bytepluses.com/api/coding</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Anthropic-compatible Coding Plan upstream.<br>
        e.g. <code style="color:#67e8f9">bytedance:bytedance-seed-code</code>
      </div>
    </div>

    <!-- Vertex AI Provider -->
    <div class="provider-card" style="border-color:{ctx.vertex_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔷 Vertex AI</h3>
        <span class="provider-status" style="background:{ctx.vertex_color}20;color:{ctx.vertex_color}">{ctx.vertex_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">vertex:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">aiplatform.googleapis.com (OpenAI-compatible)</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Google Cloud Vertex AI Gemini Enterprise Agent Platform.<br>
        e.g. <code style="color:#67e8f9">vertex:gemini-3.5-flash</code>, <code style="color:#67e8f9">vertex:gemini-3.1-pro</code>
      </div>
    </div>

    <!-- Command Code Provider -->
    <div class="provider-card" style="border-color:{ctx.command_code_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">⚡ Command Code</h3>
        <span class="provider-status" style="background:{ctx.command_code_color}20;color:{ctx.command_code_color}">{ctx.command_code_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">command_code:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.commandcode.ai/provider/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        OpenAI-compatible Provider API. Pro plan or higher required.<br>
        e.g. <code style="color:#67e8f9">command_code:deepseek-v4-flash</code>
      </div>
      <div style="margin-top:8px;font-size:0.8rem">
        <span style="color:#9ca3af;font-weight:600">Models:</span>
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">deepseek-v4-flash</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">deepseek-v4-pro</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">qwen3.6-max-preview</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">qwen3.6-plus</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">qwen3.7-max</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">kimi-k2.6</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">kimi-k2.5</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">glm-5.1</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">glm-5</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">minimax-m2.7</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">minimax-m2.5</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">step-3.5-flash</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">gemini-3.5-flash</code>
          <code style="color:#67e8f9;background:#0f172a;padding:2px 6px;border-radius:4px">gemini-3.1-flash-lite</code>
        </div>
      </div>
      </div>
    </div>
  </div>

  <!-- Ollama Cloud Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🦙 Ollama Cloud Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="ollamaKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOllamaKey()" style="margin-top:0">Save</button>
      <span id="ollamaStatus" style="color:{ctx.ollama_color}; font-size:0.85rem; font-weight:600">{ctx.ollama_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Leave empty to use local Ollama at <code style="color:#67e8f9">localhost:11434</code>
    </div>
  </div>

  <!-- Fireworks Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔥 Fireworks API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="fireworksKey" placeholder="API key (fw_...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveFireworksKey()" style="margin-top:0">Save</button>
      <span id="fireworksStatus" style="color:{ctx.fireworks_color}; font-size:0.85rem; font-weight:600">{ctx.fireworks_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">console.fireworks.ai</code>
    </div>
  </div>

  <!-- NVIDIA NIM Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟢 NVIDIA NIM API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="nvidiaKey" placeholder="API key (nvapi-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveNvidiaKey()" style="margin-top:0">Save</button>
      <span id="nvidiaStatus" style="color:{ctx.nvidia_color}; font-size:0.85rem; font-weight:600">{ctx.nvidia_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">build.nvidia.com</code>
    </div>
  </div>

  <!-- ByteDance Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔵 ByteDance API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="bytedanceKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveByteDanceKey()" style="margin-top:0">Save</button>
      <span id="bytedanceStatus" style="color:{ctx.bytedance_color}; font-size:0.85rem; font-weight:600">{ctx.bytedance_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Coding Plan API key from <code style="color:#67e8f9">console.volcengine.com/ark</code>
    </div>
  </div>

  <!-- Vertex AI / Gemini API Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔷 Gemini Agent Platform (Google Cloud)</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="vertexKey" placeholder="API key from Google Cloud Console"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveVertexKey()" style="margin-top:0">Save</button>
      <span id="vertexStatus" style="color:{ctx.vertex_color}; font-size:0.85rem; font-weight:600">{ctx.vertex_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Create in <a href="https://console.cloud.google.com/apis/credentials" target="_blank" style="color:#67e8f9">Google Cloud Console → APIs & Services → Credentials</a><br>
      Uses <code style="color:#67e8f9">google-genai</code> SDK with API key auth. No project ID needed.
    </div>
  </div>

  <!-- Command Code CLI Auth -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Command Code CLI</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <span id="commandCodeStatus" style="color:{ctx.command_code_color}; font-size:0.85rem; font-weight:600">{ctx.command_code_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Auth stored at <code style="color:#67e8f9">~/.commandcode/auth.json</code>.<br>
      Run <code style="color:#67e8f9">cmd login</code> in your terminal to authenticate.
    </div>
  </div>

  <!-- Z.AI Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Z.AI API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="zaiKey" placeholder="API key ({{ID}}.{{secret}})"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveZaiKey()" style="margin-top:0">Save</button>
      <span id="zaiStatus" style="color:{ctx.zai_color}; font-size:0.85rem; font-weight:600">{ctx.zai_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Coding plan key from <code style="color:#67e8f9">z.ai/manage-apikey/apikey-list</code>
    </div>
  </div>

  <!-- Xiaomi MiMo Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟠 Xiaomi MiMo API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="xiaomiKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveXiaomiKey()" style="margin-top:0">Save</button>
      <span id="xiaomiStatus" style="color:{ctx.xiaomi_color}; font-size:0.85rem; font-weight:600">{ctx.xiaomi_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Token plan key from <code style="color:#67e8f9">platform.xiaomimimo.com</code> — SGP endpoint
    </div>
  </div>

  <!-- OpenCode Zen Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Zen API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeKey()" style="margin-top:0">Save</button>
      <span id="opencodeStatus" style="color:{ctx.opencode_color}; font-size:0.85rem; font-weight:600">{ctx.opencode_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
    </div>
  </div>

  <!-- OpenCode Go Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Go API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeGoKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeGoKey()" style="margin-top:0">Save</button>
      <span id="opencodeGoStatus" style="color:{ctx.opencode_go_status_color}; font-size:0.85rem; font-weight:600">{ctx.opencode_go_status_text}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Go plan key — auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
    </div>
  </div>

  <!-- OpenRouter -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔵 OpenRouter</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="orKey" placeholder="API key (sk-or-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOrKey()" style="margin-top:0">Save</button>
      <span id="orStatus" style="color:{ctx.or_color}; font-size:0.85rem; font-weight:600">{ctx.or_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Use <code style="color:#67e8f9">provider/model</code> IDs (e.g. <code style="color:#67e8f9">openai/gpt-4o</code>)
    </div>
  </div>

  <!-- Kilo Gateway -->
  <div class="provider-card" style="border-color:{ctx.kilo_color}40">
    <div class="provider-header">
      <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟡 Kilo Gateway</h3>
      <span class="provider-status" style="background:{ctx.kilo_color}20;color:{ctx.kilo_color}">{ctx.kilo_status}</span>
    </div>
    <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">kilo:model-name</span></div>
    <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.kilo.ai/api/gateway</span></div>
    <div style="color:#64748b;font-size:0.8rem">
      OpenAI-compatible gateway with auto tiers and raw provider/model IDs.<br>
      e.g. <code style="color:#67e8f9">kilo:kilo-auto/balanced</code>, <code style="color:#67e8f9">kilo:anthropic/claude-sonnet-4.6</code>
    </div>
  </div>

  <!-- Kilo Gateway Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟡 Kilo Gateway API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="kiloKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveKiloKey()" style="margin-top:0">Save</button>
      <span id="kiloStatus" style="color:{ctx.kilo_color}; font-size:0.85rem; font-weight:600">{ctx.kilo_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Gateway API key from <code style="color:#67e8f9">app.kilo.ai/profile</code>
    </div>
  </div>

  <!-- Timeout Routing -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Timeout Routing <span style="font-size:0.75rem;font-weight:400;color:#64748b">(streaming only)</span></h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      <b>TTFT cutoff</b>: if no content token arrives in time, the stream is cancelled and re-routed to the most reliable model.
      <b>Max total</b>: hard wall-clock ceiling for the whole stream — anything past this is cut mid-generation with <code>[DONE]</code>.
      Non-streaming (direct) calls are NOT bounded — memory/summary workflows need the headroom.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:8px;align-items:center;cursor:pointer">
        <input type="checkbox" id="timeoutRoutingToggle" {ctx.timeout_routing_checked}
               onchange="setTimeoutRouting(this.checked)"
               style="width:16px;height:16px;cursor:pointer">
        <span style="color:#f1f5f9;font-size:0.9rem">Enabled</span>
      </label>
      <span id="timeoutRoutingStatus" style="color:{ctx.timeout_routing_color};font-size:0.85rem;font-weight:600">
        {ctx.timeout_routing_label}
      </span>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">TTFT cutoff:</span>
        <input type="number" id="timeoutCutoffInput" value="{ctx.timeout_cutoff_seconds}" min="1" max="30" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setTimeoutCutoff()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Max total:</span>
        <input type="number" id="maxTotalInput" value="{ctx.max_total_seconds}" min="1" max="60" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setMaxTotal()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <button onclick="loadRoutingStats()" style="margin-top:0;background:#334155">View Stats</button>
      <button onclick="loadRequestStats()" style="margin-top:0;background:#334155">Request Stats</button>
    </div>
    <pre id="routingStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
    <pre id="requestStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
  </div>

  <!-- Retry Policy -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔁 Retry Policy</h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      Retries on 429/5xx and transient network errors. <code>0</code> disables retry; max <code>10</code>.
      Applies to OAI-compatible upstreams (OpenRouter, Ollama, Z.AI, Xiaomi, OpenCode, Qwen, Fireworks, NVIDIA NIM).
      Claude/Codex/Antigravity have their own mitigations and are not wrapped.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Max retries (0–10):</span>
        <input type="number" id="maxRetriesInput" value="{ctx.max_retries}" min="0" max="10" step="1"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <button onclick="setMaxRetries()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Save</button>
      </label>
      <span id="maxRetriesStatus" style="color:#64748b;font-size:0.85rem;font-weight:600"></span>
    </div>
  </div>

  <!-- Reasoning Override -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🧠 Reasoning Override</h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      Forces thinking/reasoning ON at the chosen level, overriding per-request settings.
      Useful for clients like CCS (Claude Code Switcher) whose per-tier reasoning defaults
      never reach this proxy, leaving every request silently downgraded to <code>thinking=disabled</code>.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:8px;align-items:center;cursor:pointer">
        <input type="checkbox" id="reasoningOverrideToggle" {ctx.reasoning_override_checked}
               onchange="setReasoningOverride(this.checked)"
               style="width:16px;height:16px;cursor:pointer">
        <span style="color:#f1f5f9;font-size:0.9rem">Force reasoning on</span>
      </label>
      <span id="reasoningOverrideStatus" style="color:{ctx.reasoning_override_color};font-size:0.85rem;font-weight:600">
        {ctx.reasoning_override_label}
      </span>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Level:</span>
        <select id="reasoningLevelSelect"
                style="background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 8px;border-radius:4px;font-size:0.85rem">
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
        <button onclick="setReasoningLevel()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Save</button>
      </label>
      <span id="reasoningLevelStatus" style="color:#64748b;font-size:0.85rem;font-weight:600"></span>
    </div>
    <script>
      (function() {{
        var sel = document.getElementById('reasoningLevelSelect');
        if (sel) sel.value = "{ctx.reasoning_override_level}";
      }})();
    </script>
  </div>

  <!-- Reasoning Rewrite -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">&#x1f504; Reasoning Rewrite</h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      When enabled, if an upstream returns only reasoning tokens with no visible content,
      the proxy rewrites the reasoning into NPC dialogue via Fireworks. Uses Fireworks API credits.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:6px;align-items:center;cursor:pointer">
        <input type="checkbox" id="reasoningRewriteToggle" {ctx.reasoning_rewrite_checked}
               onchange="setReasoningRewrite(this.checked)"
               style="accent-color:#f59e0b">
        <span style="color:{ctx.reasoning_rewrite_color};font-size:0.85rem;font-weight:600">
          {ctx.reasoning_rewrite_label}
        </span>
      </label>
      <span id="reasoningRewriteStatus" style="color:#64748b;font-size:0.85rem;font-weight:600"></span>
    </div>
  </div>

  <!-- Quick Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Quick Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px; position:relative">
      <div style="position:relative">
        <input type="text" id="quickModelInput" placeholder="Type to search models..."
               style="width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                      border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; box-sizing:border-box"
               oninput="filterQuickModels(this.value)"
               onkeydown="handleQuickModelKeydown(event)"
               autocomplete="off">
        <div id="quickModelDropdown" style="display:none; position:absolute; top:100%; left:0; right:0; max-height:300px; overflow-y:auto; background:#1e293b; border:1px solid #334155; border-top:none; border-radius:0 0 6px 6px; z-index:100;">
        </div>
      </div>
      <div id="quickModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="quickCurrentModelId">-type to select a model-</span>
      </div>
    </div>

    <select id="modelSelect" style="display:none">
        <optgroup label="Claude Models">
          <option value="claude-sonnet-4-6">claude-sonnet-4-6 (default)</option>
          <option value="claude-opus-4-6">claude-opus-4-6</option>
          <option value="claude-sonnet-4-5-20250929">claude-sonnet-4-5-20250929</option>
          <option value="claude-haiku-4-5-20251001">claude-haiku-4-5-20251001</option>
        </optgroup>
        <optgroup label="Codex Models">
          <option value="gpt-6-sol">gpt-6-sol (default)</option>
          <option value="gpt-6-astra">gpt-6-astra</option>
          <option value="gpt-6-luna">gpt-6-luna</option>
          <option value="gpt-5.6-sol">gpt-5.6-sol</option>
          <option value="gpt-5.6-terra">gpt-5.6-terra</option>
          <option value="gpt-5.6-luna">gpt-5.6-luna</option>
          <option value="gpt-5.5">gpt-5.5</option>
          <option value="gpt-5.4">gpt-5.4</option>
        </optgroup>
        <optgroup label="Antigravity Models">
          <option value="antigravity-gemini-3-flash">antigravity-gemini-3-flash</option>
          <option value="antigravity-gemini-3.1-flash">antigravity-gemini-3.1-flash</option>
          <option value="antigravity-gemini-3.1-flash-lite">antigravity-gemini-3.1-flash-lite</option>
        </optgroup>
        <optgroup label="Kilo Gateway Models" id="kiloOptgroup">
          <option disabled value="">Loading available Kilo models…</option>
        </optgroup>
        <optgroup label="Ollama Models" id="ollamaOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="ByteDance Models" id="bytedanceOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Command Code Models" id="commandCodeOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Z.AI Models" id="zaiOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Xiaomi MiMo Models (V2.5 + V2.6)">
          <option value="xiaomi:mimo-v2.5-pro">xiaomi:mimo-v2.5-pro (flagship)</option>
          <option value="xiaomi:mimo-v2.5">xiaomi:mimo-v2.5 (omni-modal)</option>
          <option value="xiaomi:mimo-v2.5-asr">xiaomi:mimo-v2.5-asr (platform)</option>
          <option value="xiaomi:mimo-v2.5-tts">xiaomi:mimo-v2.5-tts (platform)</option>
          <option value="xiaomi:mimo-v2.5-tts-voiceclone">xiaomi:mimo-v2.5-tts-voiceclone (platform)</option>
          <option value="xiaomi:mimo-v2.5-tts-voicedesign">xiaomi:mimo-v2.5-tts-voicedesign (platform)</option>
          <option value="xiaomi:mimo-v2.6-flash">xiaomi:mimo-v2.6-flash (platform)</option>
          <option value="xiaomi:mimo-v2.6-pro">xiaomi:mimo-v2.6-pro</option>
        </optgroup>
        <optgroup label="OpenCode Zen Models">
          <option disabled value="">No static suggestions — Zen has no live catalog endpoint</option>
        </optgroup>
        <optgroup label="OpenCode Go Models">
          <option value="opencode-go:glm-5.3">opencode-go:glm-5.3</option>
          <option value="opencode-go:kimi-k3">opencode-go:kimi-k3</option>
          <option value="opencode-go:kimi-k2.6">opencode-go:kimi-k2.6</option>
          <option value="opencode-go:mimo-v2.6-flash">opencode-go:mimo-v2.6-flash</option>
          <option value="opencode-go:mimo-v2.5-pro">opencode-go:mimo-v2.5-pro</option>
          <option value="opencode-go:minimax-m2.5">opencode-go:minimax-m2.5</option>
          <option value="opencode-go:minimax-m2.7">opencode-go:minimax-m2.7</option>
          <option value="opencode-go:qwen3.5-plus">opencode-go:qwen3.5-plus</option>
          <option value="opencode-go:qwen3.6-plus">opencode-go:qwen3.6-plus</option>
        </optgroup>
        <optgroup label="Qwen Code Models">
          <option value="qwen:coder-model">qwen:coder-model</option>
        </optgroup>
        <optgroup label="Fireworks Models">
          <option value="fireworks:kimi-k2p5-turbo">fireworks:kimi-k2p5-turbo</option>
          <option value="fireworks:kimi-k3">fireworks:kimi-k3</option>
          <option value="fireworks:deepseek-v4-pro">fireworks:deepseek-v4-pro</option>
          <option value="fireworks:deepseek-r1">fireworks:deepseek-r1</option>
          <option value="fireworks:minimax-m2p7">fireworks:minimax-m2p7</option>
        </optgroup>
        <optgroup label="NVIDIA NIM Models (Free)">
          <option value="nvidia:kimi-k2">nvidia:kimi-k2</option>
          <option value="nvidia:kimi-k2-0905">nvidia:kimi-k2-0905</option>
          <option value="nvidia:kimi-k2-thinking">nvidia:kimi-k2-thinking</option>
          <option value="nvidia:glm-5.3">nvidia:glm-5.3</option>
          <option value="nvidia:kimi-k3">nvidia:kimi-k3</option>
          <option value="nvidia:deepseek-v3.2">nvidia:deepseek-v3.2</option>
          <option value="nvidia:deepseek-v3.1">nvidia:deepseek-v3.1</option>
          <option value="nvidia:deepseek-v3.1-terminus">nvidia:deepseek-v3.1-terminus</option>
          <option value="nvidia:deepseek-v4-flash">nvidia:deepseek-v4-flash</option>
          <option value="nvidia:deepseek-v4-pro">nvidia:deepseek-v4-pro</option>
          <option value="nvidia:step-3.5-flash">nvidia:step-3.5-flash</option>
          <option value="nvidia:minimax-m2.7">nvidia:minimax-m2.7</option>
          <option value="nvidia:llama-4-maverick">nvidia:llama-4-maverick</option>
          <option value="nvidia:mistral-large-3">nvidia:mistral-large-3</option>
          <option value="nvidia:devstral-2">nvidia:devstral-2</option>
          <option value="nvidia:magistral-small">nvidia:magistral-small</option>
          <option value="nvidia:mistral-nemotron">nvidia:mistral-nemotron</option>
          <option value="nvidia:qwen3-coder-480b">nvidia:qwen3-coder-480b</option>
          <option value="nvidia:seed-oss-36b">nvidia:seed-oss-36b</option>
          <option value="nvidia:phi-3.5-mini">nvidia:phi-3.5-mini</option>
        </optgroup>
        <optgroup label="Gemini Agent Platform (Google Cloud)">
          <option value="vertex:gemini-3.8-flash">vertex:gemini-3.8-flash</option>
          <option value="vertex:gemini-3.5-flash">vertex:gemini-3.5-flash (default)</option>
          <option value="vertex:gemini-3.1-pro">vertex:gemini-3.1-pro</option>
          <option value="vertex:gemini-3.1-flash-lite">vertex:gemini-3.1-flash-lite</option>
          <option value="vertex:gemini-3-flash">vertex:gemini-3-flash</option>
          <option value="vertex:gemini-3.5-flash-lite">vertex:gemini-3.5-flash-lite</option>
          <option value="vertex:gemini-3.6-flash">vertex:gemini-3.6-flash</option>
          <option value="vertex:gemini-3.7-flash">vertex:gemini-3.7-flash</option>
        </optgroup>
      </select>
    <textarea id="sysPrompt" rows="2" placeholder="System prompt">You are Lydia, a Nord housecarl sworn to protect the Dragonborn. Stay in character. One sentence only.</textarea>
    <textarea id="userMsg" rows="1" placeholder="User message" style="margin-top:6px">What do you think of dragons?</textarea>
    <button onclick="testChat()" id="testBtn">Send</button>
    <div id="response" style="display:none"></div>
    <div id="timing" class="timing"></div>
  </div>

  <!-- Vision Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Vision Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px; position:relative">
      <div style="position:relative">
        <input type="text" id="visionModelInput" placeholder="Type to search vision models..."
               style="width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                      border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; box-sizing:border-box"
               oninput="filterVisionModels(this.value)"
               onkeydown="handleVisionModelKeydown(event)"
               autocomplete="off">
        <div id="visionModelDropdown" style="display:none; position:absolute; top:100%; left:0; right:0; max-height:300px; overflow-y:auto; background:#1e293b; border:1px solid #334155; border-top:none; border-radius:0 0 6px 6px; z-index:100;">
        </div>
      </div>
      <div id="visionModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="visionCurrentModelId">-type to select a vision model-</span>
      </div>
    </div>

    <textarea id="visionPrompt" rows="2" placeholder="Describe what you want to know about the image">What do you see in this image? Describe it briefly.</textarea>

    <!-- Image Selection -->
    <div style="margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155">
      <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px">Test Image:</div>
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="random" checked
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Random test image</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="upload"
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Upload image</span>
        </label>
      </div>

      <div id="visionRandomPreview" style="margin-top:10px">
        <div style="width:120px; height:120px; border:2px dashed #334155; border-radius:6px; display:flex; align-items:center; justify-content:center; background:#1e293b; position:relative; overflow:hidden">
          <canvas id="visionRandomCanvas" width="120" height="120"></canvas>
        </div>
        <button onclick="generateRandomVisionImage()" type="button" style="margin-top:8px; background:#334155; padding:6px 12px; font-size:0.8rem">Generate new image</button>
      </div>

      <div id="visionUploadSection" style="display:none; margin-top:10px">
        <input type="file" id="visionImageFile" accept="image/*"
               style="color:#e2e8f0; font-size:0.85rem">
        <div style="color:#64748b; font-size:0.75rem; margin-top:6px">Supports: PNG, JPG, GIF, WebP (max 4MB)</div>
      </div>
    </div>

    <button onclick="testVision()" id="visionTestBtn" style="margin-top:12px">Analyze Image</button>
    <div id="visionResponse" style="display:none; margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap"></div>
    <div id="visionTiming" class="timing"></div>
  </div>

<script>
async function saveOrKey() {{
  const key = document.getElementById('orKey').value.trim();
  const status = document.getElementById('orStatus');
  try {{
    const r = await fetch('/config/openrouter-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Saved' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('orKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveKiloKey() {{
  const key = document.getElementById('kiloKey').value.trim();
  const status = document.getElementById('kiloStatus');
  try {{
    const r = await fetch('/config/kilo-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('kiloKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveOllamaKey() {{
  const key = document.getElementById('ollamaKey').value.trim();
  const status = document.getElementById('ollamaStatus');
  try {{
    const r = await fetch('/config/ollama-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Cloud (key configured)' : 'Local (localhost:11434)';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('ollamaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveFireworksKey() {{
  const key = document.getElementById('fireworksKey').value.trim();
  const status = document.getElementById('fireworksStatus');
  try {{
    const r = await fetch('/config/fireworks-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('fireworksKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveNvidiaKey() {{
  const key = document.getElementById('nvidiaKey').value.trim();
  const status = document.getElementById('nvidiaStatus');
  try {{
    const r = await fetch('/config/nvidia-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('nvidiaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveByteDanceKey() {{
  const key = document.getElementById('bytedanceKey').value.trim();
  const status = document.getElementById('bytedanceStatus');
  try {{
    const r = await fetch('/config/bytedance-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('bytedanceKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveVertexKey() {{
  const key = document.getElementById('vertexKey').value.trim();
  const status = document.getElementById('vertexStatus');
  try {{
    const r = await fetch('/config/vertex-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('vertexKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveZaiKey() {{
  const key = document.getElementById('zaiKey').value.trim();
  const status = document.getElementById('zaiStatus');
  try {{
    const r = await fetch('/config/zai-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('zaiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveXiaomiKey() {{
  const key = document.getElementById('xiaomiKey').value.trim();
  const status = document.getElementById('xiaomiStatus');
  try {{
    const r = await fetch('/config/xiaomi-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('xiaomiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeKey() {{
  const key = document.getElementById('opencodeKey').value.trim();
  const status = document.getElementById('opencodeStatus');
  try {{
    const r = await fetch('/config/opencode-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeGoKey() {{
  const key = document.getElementById('opencodeGoKey').value.trim();
  const status = document.getElementById('opencodeGoStatus');
  try {{
    const r = await fetch('/config/opencode-go-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    if (!r.ok) throw new Error('Server error ' + r.status);
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeGoKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function testChat() {{
  const btn = document.getElementById('testBtn');
  const resp = document.getElementById('response');
  const timing = document.getElementById('timing');

  let model = document.getElementById('quickModelInput').value.trim();

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

  btn.disabled = true; btn.textContent = 'Waiting...';
  resp.style.display = 'block'; resp.textContent = '...';
  timing.textContent = '';
  const start = Date.now();
  try {{
    const r = await fetch('/v1/chat/completions', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'system', content: document.getElementById('sysPrompt').value}},
          {{role: 'user', content: document.getElementById('userMsg').value}}
        ]
      }})
    }});
    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    if (data.choices) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
  }}
  btn.disabled = false; btn.textContent = 'Send';
}}

async function loadOllamaModels() {{
  const group = document.getElementById('ollamaOptgroup');
  try {{
    const r = await fetch('/config/ollama-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'ollama:' + m;
        opt.textContent = 'ollama:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No local models (run: ollama pull <model>)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Ollama not running</option>';
  }}
  _initQuickModels();
}}
async function loadByteDanceModels() {{
  const group = document.getElementById('bytedanceOptgroup');
  var _bdAllowPrefixes = ['seed-2-0-pro', 'seed-2-0-lite', 'seed-2-0-code', 'glm-4-7', 'kimi-k2-', 'gpt-oss-120b'];
  var _bdAliases = {{'kimi-k2-250905': 'kimi-k2.5', 'kimi-k2-250711': 'kimi-k2.5'}};
  try {{
    const r = await fetch('/config/bytedance-models');
    const data = await r.json();
    group.innerHTML = '';
    const allowed = (data.models || []).filter(function(m) {{
      return _bdAllowPrefixes.some(function(p) {{ return m.startsWith(p); }});
    }});
    // Deduplicate after aliasing (multiple kimi versions collapse to one)
    var seen = {{}};
    var deduped = [];
    allowed.forEach(function(m) {{
      var display = _bdAliases[m] || m;
      if (!seen[display]) {{ seen[display] = true; deduped.push({{raw: m, display: display}}); }}
    }});
    if (deduped.length > 0) {{
      deduped.forEach(function(item) {{
        const opt = document.createElement('option');
        opt.value = 'bytedance:' + item.raw;
        opt.textContent = 'bytedance:' + item.display;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No ByteDance models (check API key)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">ByteDance unavailable</option>';
  }}
  _initQuickModels();
}}
async function loadCommandCodeModels() {{
  const group = document.getElementById('commandCodeOptgroup');
  try {{
    const r = await fetch('/config/command-code-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'command_code:' + m;
        opt.textContent = 'command_code:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No models available';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Command Code unavailable</option>';
  }}
  _initQuickModels();
}}


async function loadZaiModels() {{
  const group = document.getElementById('zaiOptgroup');
  try {{
    const r = await fetch('/config/zai-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'zai:' + m;
        opt.textContent = 'zai:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No Z.AI models (check API key)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Z.AI unavailable</option>';
  }}
  _initQuickModels();
}}
async function loadKiloModels() {{
  const group = document.getElementById('kiloOptgroup');
  try {{
    const r = await fetch('/config/kilo-models');
    const data = await r.json();
    group.innerHTML = '';
    const freeModels = (data.models || []).filter(function(m) {{ return m.indexOf('free') !== -1; }});
    if (freeModels.length > 0) {{
      freeModels.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No free Kilo models available';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Kilo unavailable</option>';
  }}
  _initQuickModels();
}}
document.addEventListener('DOMContentLoaded', loadOllamaModels);
document.addEventListener('DOMContentLoaded', loadByteDanceModels);
document.addEventListener('DOMContentLoaded', loadCommandCodeModels);
document.addEventListener('DOMContentLoaded', loadZaiModels);
document.addEventListener('DOMContentLoaded', loadKiloModels);
document.addEventListener('DOMContentLoaded', loadVisionModels);
document.addEventListener('DOMContentLoaded', function() {{
  // Initialize vision test UI
  generateRandomVisionImage();
  toggleVisionImageSource();
}});

var _allVisionModels = [];
var _visionModelIndex = -1;

async function loadVisionModels() {{
  try {{
    const r = await fetch('/config/vision-models');
    const data = await r.json();
    _allVisionModels = [];
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        _allVisionModels.push({{value: m.id, label: m.name || m.id}});
      }});
    }}
  }} catch(e) {{
    _allVisionModels = [];
    console.error('Failed to load vision models:', e);
  }}
}}

function filterVisionModels(query) {{
  var dropdown = document.getElementById('visionModelDropdown');
  var q = query.trim().toLowerCase();
  var matches;
  if (!q) {{
    matches = _allVisionModels;
  }} else {{
    matches = _allVisionModels.filter(function(m) {{
      return m.value.toLowerCase().indexOf(q) !== -1 || m.label.toLowerCase().indexOf(q) !== -1;
    }});
  }}
  if (matches.length === 0) {{
    dropdown.innerHTML = '<div style="padding:8px 12px;color:#64748b;font-size:0.85rem">No matches</div>';
    dropdown.style.display = 'block';
    _visionModelIndex = -1;
    return;
  }}
  var html = matches.map(function(m, i) {{
    return '<div class="vision-model-item" data-value="' + m.value.replace(/"/g, '&quot;') + '" data-index="' + i + '" style="padding:6px 12px;cursor:pointer;font-size:0.85rem;font-family:monospace;color:#e2e8f0;border-bottom:1px solid #1e293b" onmousedown="selectVisionModel(this.dataset.value)">' +
           '<span style="color:#67e8f9">' + m.value.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span>' +
           (m.label !== m.value ? ' <span style="color:#64748b">— ' + m.label.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span>' : '') +
           '</div>';
  }}).join('');
  dropdown.innerHTML = html;
  dropdown.style.display = 'block';
  _visionModelIndex = -1;
}}

function handleVisionModelKeydown(e) {{
  var items = document.querySelectorAll('.vision-model-item');
  if (e.key === 'ArrowDown') {{
    e.preventDefault();
    _visionModelIndex = Math.min(_visionModelIndex + 1, items.length - 1);
    _updateVisionModelHighlight(items);
  }} else if (e.key === 'ArrowUp') {{
    e.preventDefault();
    _visionModelIndex = Math.max(_visionModelIndex - 1, -1);
    _updateVisionModelHighlight(items);
  }} else if (e.key === 'Enter') {{
    e.preventDefault();
    if (_visionModelIndex >= 0 && items[_visionModelIndex]) {{
      selectVisionModel(items[_visionModelIndex].dataset.value);
    }} else if (items.length > 0) {{
      var val = document.getElementById('visionModelInput').value.trim();
      var exact = _allVisionModels.find(function(m) {{ return m.value === val; }});
      if (exact) {{
        selectVisionModel(exact.value);
      }} else {{
        selectVisionModel(items[0].dataset.value);
      }}
    }}
  }} else if (e.key === 'Escape') {{
    document.getElementById('visionModelDropdown').style.display = 'none';
    _visionModelIndex = -1;
  }}
}}

function _updateVisionModelHighlight(items) {{
  items.forEach(function(item, i) {{
    item.style.background = i === _visionModelIndex ? '#334155' : 'transparent';
  }});
  if (_visionModelIndex >= 0 && items[_visionModelIndex]) {{
    items[_visionModelIndex].scrollIntoView({{block: 'nearest'}});
  }}
}}

function selectVisionModel(value) {{
  document.getElementById('visionModelInput').value = value;
  document.getElementById('visionCurrentModelId').textContent = value;
  document.getElementById('visionModelDropdown').style.display = 'none';
  _visionModelIndex = -1;
}}

document.addEventListener('DOMContentLoaded', function() {{
  var vInput = document.getElementById('visionModelInput');
  if (vInput) {{
    vInput.addEventListener('blur', function() {{
      setTimeout(function() {{ document.getElementById('visionModelDropdown').style.display='none'; }}, 150);
    }});
    vInput.addEventListener('focus', function() {{
      filterVisionModels(vInput.value);
    }});
  }}
}});

var _allQuickModels = [];
var _quickModelIndex = -1;

function _initQuickModels() {{
  _allQuickModels = [];
  var select = document.getElementById('modelSelect');
  if (select) {{
    select.querySelectorAll('option').forEach(function(opt) {{
      if (!opt.disabled && opt.value) {{
        _allQuickModels.push({{value: opt.value, label: opt.textContent || opt.value}});
      }}
    }});
  }}
}}

function filterQuickModels(query) {{
  var dropdown = document.getElementById('quickModelDropdown');
  var q = query.trim().toLowerCase();
  var matches;
  if (!q) {{
    matches = _allQuickModels;
  }} else {{
    matches = _allQuickModels.filter(function(m) {{
      return m.value.toLowerCase().indexOf(q) !== -1 || m.label.toLowerCase().indexOf(q) !== -1;
    }});
  }}
  if (matches.length === 0) {{
    dropdown.innerHTML = '<div style="padding:8px 12px;color:#64748b;font-size:0.85rem">No matches</div>';
    dropdown.style.display = 'block';
    _quickModelIndex = -1;
    return;
  }}
  var html = matches.map(function(m, i) {{
    return '<div class="quick-model-item" data-value="' + m.value.replace(/"/g, '&quot;') + '" data-index="' + i + '" style="padding:6px 12px;cursor:pointer;font-size:0.85rem;font-family:monospace;color:#e2e8f0;border-bottom:1px solid #1e293b" onmousedown="selectQuickModel(this.dataset.value)">' +
           '<span style="color:#67e8f9">' + m.value.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span>' +
           (m.label !== m.value ? ' <span style="color:#64748b">— ' + m.label.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span>' : '') +
           '</div>';
  }}).join('');
  dropdown.innerHTML = html;
  dropdown.style.display = 'block';
  _quickModelIndex = -1;
}}

function handleQuickModelKeydown(e) {{
  var items = document.querySelectorAll('.quick-model-item');
  if (e.key === 'ArrowDown') {{
    e.preventDefault();
    _quickModelIndex = Math.min(_quickModelIndex + 1, items.length - 1);
    _updateQuickModelHighlight(items);
  }} else if (e.key === 'ArrowUp') {{
    e.preventDefault();
    _quickModelIndex = Math.max(_quickModelIndex - 1, -1);
    _updateQuickModelHighlight(items);
  }} else if (e.key === 'Enter') {{
    e.preventDefault();
    if (_quickModelIndex >= 0 && items[_quickModelIndex]) {{
      selectQuickModel(items[_quickModelIndex].dataset.value);
    }} else if (items.length > 0) {{
      var val = document.getElementById('quickModelInput').value.trim();
      var exact = _allQuickModels.find(function(m) {{ return m.value === val; }});
      if (exact) {{
        selectQuickModel(exact.value);
      }} else {{
        selectQuickModel(items[0].dataset.value);
      }}
    }}
  }} else if (e.key === 'Escape') {{
    document.getElementById('quickModelDropdown').style.display = 'none';
    _quickModelIndex = -1;
  }}
}}

function _updateQuickModelHighlight(items) {{
  items.forEach(function(item, i) {{
    item.style.background = i === _quickModelIndex ? '#334155' : 'transparent';
  }});
  if (_quickModelIndex >= 0 && items[_quickModelIndex]) {{
    items[_quickModelIndex].scrollIntoView({{block: 'nearest'}});
  }}
}}

function selectQuickModel(value) {{
  document.getElementById('quickModelInput').value = value;
  document.getElementById('quickCurrentModelId').textContent = value;
  document.getElementById('quickModelDropdown').style.display = 'none';
  _quickModelIndex = -1;
}}

document.addEventListener('DOMContentLoaded', function() {{
  _initQuickModels();
  const input = document.getElementById('quickModelInput');
  if (input) {{
    input.addEventListener('blur', function() {{
      setTimeout(function() {{ document.getElementById('quickModelDropdown').style.display='none'; }}, 150);
    }});
    input.addEventListener('focus', function() {{
      filterQuickModels(input.value);
    }});
  }}
}});

function toggleVisionImageSource() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;
  const randomSection = document.getElementById('visionRandomPreview');
  const uploadSection = document.getElementById('visionUploadSection');

  if (source === 'random') {{
    randomSection.style.display = 'block';
    uploadSection.style.display = 'none';
    generateRandomVisionImage();
  }} else {{
    randomSection.style.display = 'none';
    uploadSection.style.display = 'block';
  }}
}}

function generateRandomVisionImage() {{
  const canvas = document.getElementById('visionRandomCanvas');
  const ctx = canvas.getContext('2d');

  // Generate random colors
  const hue1 = Math.floor(Math.random() * 360);
  const hue2 = (hue1 + 120 + Math.floor(Math.random() * 120)) % 360;
  const hue3 = (hue2 + 120 + Math.floor(Math.random() * 120)) % 360;

  // Fill background
  ctx.fillStyle = `hsl(${{hue1}}, 60%, 20%)`;
  ctx.fillRect(0, 0, 120, 120);

  // Draw random shapes
  for (let i = 0; i < 8; i++) {{
    ctx.fillStyle = `hsl(${{Math.random() * 360}}, 70%, ${{40 + Math.random() * 40}}%)`;
    const x = Math.random() * 120;
    const y = Math.random() * 120;
    const size = 10 + Math.random() * 40;

    if (Math.random() > 0.5) {{
      // Circle
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    }} else {{
      // Rectangle
      ctx.fillRect(x - size/2, y - size/2, size, size);
    }}
  }}

  // Add some lines
  for (let i = 0; i < 5; i++) {{
    ctx.strokeStyle = `hsl(${{Math.random() * 360}}, 80%, 60%)`;
    ctx.lineWidth = 2 + Math.random() * 4;
    ctx.beginPath();
    ctx.moveTo(Math.random() * 120, Math.random() * 120);
    ctx.lineTo(Math.random() * 120, Math.random() * 120);
    ctx.stroke();
  }}

  // Store the base64 for later use
  canvas.dataset.base64 = canvas.toDataURL('image/png');
}}

function getVisionImageBase64() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;

  if (source === 'random') {{
    const canvas = document.getElementById('visionRandomCanvas');
    return canvas.dataset.base64 || canvas.toDataURL('image/png');
  }} else {{
    const fileInput = document.getElementById('visionImageFile');
    return new Promise((resolve, reject) => {{
      if (!fileInput.files || fileInput.files.length === 0) {{
        reject(new Error('Please select an image file'));
        return;
      }}
      const file = fileInput.files[0];
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Error reading image file'));
      reader.readAsDataURL(file);
    }});
  }}
}}

async function testVision() {{
  const btn = document.getElementById('visionTestBtn');
  const resp = document.getElementById('visionResponse');
  const timing = document.getElementById('visionTiming');

  let model = document.getElementById('visionModelInput').value.trim();

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

  // Check image source
  const imageSource = document.querySelector('input[name="visionImageSource"]:checked').value;
  if (imageSource === 'upload') {{
    const fileInput = document.getElementById('visionImageFile');
    if (!fileInput.files || fileInput.files.length === 0) {{
      resp.style.display = 'block';
      resp.textContent = 'Please select an image file';
      return;
    }}
  }}

  btn.disabled = true; btn.textContent = 'Analyzing...';
  resp.style.display = 'block'; resp.textContent = 'Processing image...';
  timing.textContent = '';

  const start = Date.now();

  try {{
    // Get image base64
    let base64Image;
    if (imageSource === 'random') {{
      base64Image = getVisionImageBase64();
    }} else {{
      base64Image = await getVisionImageBase64();
    }}

    const prompt = document.getElementById('visionPrompt').value || 'What do you see in this image?';

    const r = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'user', content: [
            {{type: 'text', text: prompt}},
            {{type: 'image_url', image_url: {{url: base64Image}}}}
          ]}}
        ]
      }})
    }});

    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (data.choices && data.choices[0] && data.choices[0].message) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else if (data.error) {{
      resp.textContent = 'Error: ' + (data.error.message || JSON.stringify(data.error));
      timing.textContent = '';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
    timing.textContent = '';
  }}

  btn.disabled = false; btn.textContent = 'Analyze Image';
}}

async function setTimeoutRouting(enabled) {{
  const status = document.getElementById('timeoutRoutingStatus');
  try {{
    const resp = await fetch('/config/timeout-routing', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{enabled}}),
    }});
    const data = await resp.json();
    status.textContent = data.timeout_routing_enabled ? 'Active' : 'Disabled';
    status.style.color = data.timeout_routing_enabled ? '#4ade80' : '#64748b';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
  }}
}}

async function loadRoutingStats() {{
  const pre = document.getElementById('routingStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/config/timeout-routing');
    const data = await resp.json();
    if (Object.keys(data.stats).length === 0) {{
      pre.textContent = 'No data yet — stats accumulate as streaming requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data.stats, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
}}

async function setTimeoutCutoff() {{
  const input = document.getElementById('timeoutCutoffInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 30) {{
    alert('TTFT cutoff must be between 1 and 30 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/timeout-cutoff', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.timeout_cutoff_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function setMaxTotal() {{
  const input = document.getElementById('maxTotalInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 60) {{
    alert('Max total must be between 1 and 60 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/max-total', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.max_total_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function setMaxRetries() {{
  const input = document.getElementById('maxRetriesInput');
  const status = document.getElementById('maxRetriesStatus');
  const n = parseInt(input.value, 10);
  if (isNaN(n) || n < 0 || n > 10) {{
    status.textContent = 'Must be 0..10';
    status.style.color = '#f87171';
    return;
  }}
  try {{
    const resp = await fetch('/config/max-retries', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{max_retries: n}}),
    }});
    if (resp.ok) {{
      const data = await resp.json();
      input.value = data.max_retries;
      status.textContent = 'Saved (' + data.max_retries + ')';
      status.style.color = '#4ade80';
    }} else {{
      const err = await resp.json();
      status.textContent = 'Error: ' + (err.detail || resp.status);
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function setReasoningOverride(enabled) {{
  const status = document.getElementById('reasoningOverrideStatus');
  try {{
    const resp = await fetch('/config/reasoning-override', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{enabled}}),
    }});
    if (resp.ok) {{
      const data = await resp.json();
      status.textContent = data.reasoning_override_enabled ? 'Active (' + data.reasoning_override_level + ')' : 'Disabled';
      status.style.color = data.reasoning_override_enabled ? '#4ade80' : '#64748b';
    }} else {{
      const err = await resp.json();
      status.textContent = 'Error: ' + (err.detail || resp.status);
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function setReasoningLevel() {{
  const sel = document.getElementById('reasoningLevelSelect');
  const status = document.getElementById('reasoningLevelStatus');
  const toggleStatus = document.getElementById('reasoningOverrideStatus');
  const level = sel.value;
  if (level !== 'low' && level !== 'medium' && level !== 'high') {{
    status.textContent = 'Invalid level';
    status.style.color = '#f87171';
    return;
  }}
  try {{
    const resp = await fetch('/config/reasoning-level', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{level}}),
    }});
    if (resp.ok) {{
      const data = await resp.json();
      status.textContent = 'Saved (' + data.reasoning_override_level + ')';
      status.style.color = '#4ade80';
      if (data.reasoning_override_enabled) {{
        toggleStatus.textContent = 'Active (' + data.reasoning_override_level + ')';
        toggleStatus.style.color = '#4ade80';
      }}
    }} else {{
      const err = await resp.json();
      status.textContent = 'Error: ' + (err.detail || resp.status);
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function setReasoningRewrite(enabled) {{
  const status = document.getElementById('reasoningRewriteStatus');
  try {{
    const resp = await fetch('/config/reasoning-rewrite', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{enabled}}),
    }});
    if (resp.ok) {{
      const data = await resp.json();
      status.textContent = data.status === 'ok' ? '\u2713 Saved' : '\u2717 Error';
      status.style.color = data.status === 'ok' ? '#4ade80' : '#f87171';
      setTimeout(() => status.textContent = '', 2000);
    }} else {{
      const err = await resp.json();
      status.textContent = '\u2717 ' + (err.detail || resp.status);
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = '\u2717 ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function loadRequestStats() {{
  const pre = document.getElementById('requestStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/stats');
    const data = await resp.json();
    if (Object.keys(data.by_model).length === 0) {{
      pre.textContent = 'No request data yet — stats accumulate as requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
}}

async function removeAccount(email) {{
  if (!confirm('Remove account ' + email + '?')) return;
  try {{
    const resp = await fetch('/config/antigravity-account/' + encodeURIComponent(email), {{
      method: 'DELETE'
    }});
    const data = await resp.json();
    if (data.status === 'removed') {{
      alert('Account removed: ' + email);
      location.reload();
    }} else {{
      alert('Account not found: ' + email);
    }}
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}
</script>
</body></html>"""
