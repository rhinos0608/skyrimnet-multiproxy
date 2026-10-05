# Technical documentation

Build instructions, architecture, release packaging, and the `proxy.py` patch internals for
SkyrimNet MultiProxy. See `README.md` for what it does in-game.

## Prerequisites

- CMake 3.24+
- MSVC Build Tools (VS 2022+) — the C++ compiler, full IDE not required
- Internet access on the first build only — CMake's `FetchContent` pulls CommonLibSSE-NG, fmt,
  spdlog, and rapidcsv automatically

## Building

Open an **x64 Native Tools Command Prompt for VS 2022**, then:

```cmd
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_INSTALL_RULES=ON
cmake --build build --config Release
```

The first build takes several minutes (compiling CommonLibSSE-NG); later builds only recompile
changed files. The compiled DLL lands in `build/Release/SkyrimNetMultiProxy.dll`.

The build does **not** touch your live game install by default — test with a real
Vortex/MO2-style package install instead of a raw file copy. If you specifically want the old
convenience behavior (auto-copy the DLL into your game's `Data/SKSE/Plugins/` after every build),
opt in explicitly, supplying your own install path (never hardcoded/committed):

```cmd
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_INSTALL_RULES=ON -DDEPLOY_TO_GAME=ON -DSKYRIM_PATH="C:/path/to/Skyrim Special Edition"
```

## Versioning

`CMakeLists.txt`'s `project(... VERSION X.Y.Z ...)` is the one place a version bump happens.
Everything else reads from it — there is no second literal to remember:

- `SKYRIMNET_MULTIPROXY_VERSION_STRING` (`main.cpp`'s startup log line) is `PROJECT_VERSION` as a
  string.
- `SKSEPluginInfo`'s `.Version` field (what SKSE, crash logs, and mod managers report for the
  plugin itself) is built from `PROJECT_VERSION_MAJOR`/`_MINOR`/`_PATCH`, the same components
  `project()` derives from that one `VERSION X.Y.Z`.

Both used to be separate hardcoded literals. The log string was fixed to track `PROJECT_VERSION` in
the v2.3.0 bump; `.Version` was missed and stayed at `2.0.0.0` through v2.1.0-v2.3.0 despite three
real version bumps, silently telling anyone reading a crash log or a mod manager's plugin list the
wrong version. Fixed in the v2.4.0 cycle by wiring `.Version` to the same `PROJECT_VERSION`
components rather than hand-writing a fourth copy of the number — bump `CMakeLists.txt` once and
both move together, confirmed by temporarily building against a fake version and reading the
result back out of the compiled DLL (`SKSEPlugin_Query`'s reported `version`, not just the log
string) rather than trusting the build succeeded.

## Architecture

On `SKSEPlugin_Load` (before the main menu), the plugin:

1. Reads `Data\SKSE\Plugins\SkyrimNetMultiProxy.ini` — or, if that doesn't exist yet but the
   pre-rename `ProxyLauncher.ini` does (an upgrading install), copies that forward to the new
   filename and reads it instead, so an existing install's settings keep working without
   re-running setup. Logs when this fallback fires.
2. Checks whether the configured port is already listening — a manually pre-launched proxy (or a
   still-present old `ProxyLauncher.dll` racing to launch its own copy) is detected and left alone
   rather than duplicated.
3. If nothing's listening, launches the proxy as a detached, minimized console process via
   a pure Win32 process launcher (`proxy_launcher.cpp` has no CommonLibSSE dependency — it's a
   thin wrapper Skyrim's own event system happens to trigger). The launcher passes
   `--plugin-version` (this DLL's own version, for the dashboard's match check) on the command
   line and reads the listen port from the plugin's own `Port` setting (default `8000`); the proxy
   itself takes `--port` (default `8000`) for manual launches and binds `127.0.0.1` by default —
   loopback only, never the LAN.
4. Logs the result to `Documents\My Games\Skyrim Special Edition\SKSE\SkyrimNetMultiProxy.log`.
5. If `Data\SKSE\Plugins\ProxyLauncher.dll` (the pre-rename DLL) is still present, logs a one-time
   warning that it's dead weight and safe to delete, alongside `ProxyLauncher.ini`.

The proxy warms up in the background while the game loads; the first NPC conversation waits up to
60s for auth to be ready (handled inside the proxy itself, not this plugin).

## The shipped runtime is proxy-only

The packaged runtime serves the OpenAI-compatible chat API and the local dashboard — nothing else.
There is no MCP server mode in the shipped tree (`mcp_server.py` / `orchestrator.py` are not
packaged, `--mode mcp` / `--transport` do not exist here, and `requirements.txt` carries no `mcp`
dependency). Manual launch is `python proxy.py --port 8000` from the release folder
(`start-proxy.bat` does exactly this); `--plugin-version` is normally supplied by the SKSE launcher
so the dashboard can compare proxy vs. plugin versions.

### Provider concurrency limits

`proxy_internal/concurrency.py` hands each provider a pool of simultaneous-request slots before the
routing chain calls it. This is what keeps a plan-wide cap — z.ai's coding plans allow 6 concurrent
calls — from turning into upstream 429s; a request that cannot get a slot in time fails with HTTP
503 so SkyrimNet's smart routing can move to the next provider.

Limits live in `config.json` as `providers.<name>.max_concurrent` (`0` = unlimited) and
`providers.<name>.queue_timeout_ms` (`0` = wait forever, default 60000). zai defaults to 6 with no
config entry; every other provider is unlimited until one is added. Live usage (`limit`, `active`,
`queued`) is reported per provider on `/health` under `concurrency`.

### Context-window trimming

`proxy_internal/context_trim.py` handles the upstream 400 that means "prompt too long". When a
provider reports `context_length_exceeded` with token counts, the Z.AI provider drops the oldest
droppable history — never the first message (SkyrimNet's diary/scenario context), never the final
message (the turn being answered), never a `system` message — and retries, up to `MAX_TRIM_ROUNDS`
rounds, before surfacing the error. Streaming retries are safe because the rejection arrives while
opening the stream, before any event reaches the client.

## Release packaging

Starting with this release, the distributed zip bundles a complete, ready-to-run Python runtime
directly — no separate download, no patch step for the end user. That tree (`proxy.py` plus the
full `proxy_internal/` package) is tracked in this repo; it is the Python source of truth, and
`proxy-runtime.manifest` names every `.py` file the zip may contain (no wildcards — a new module
does not ship until someone adds it to the manifest).

> **Status: UNRELEASED.** The packaged runtime tree in this checkout has not shipped in any
> release yet. Do not describe it as shipped.

A release package contains:

```
SkyrimNetMultiProxy-vX.Y.Z.zip
├── SkyrimNetMultiProxy.dll   — the SKSE plugin
├── SkyrimNetMultiProxy.ini   — SKSE plugin config template
├── proxy.py                  — runtime entry point (tracked here)
├── proxy_internal/            — runtime package, per proxy-runtime.manifest
├── requirements.txt          — runtime-only Python deps (dev packages stripped)
├── config.example.json       — API-key template (placeholders only)
├── start-proxy.bat           — manual-launch helper (not required — the plugin starts it)
├── setup.ps1 / setup.bat     — checks Python, runs pip install, prints the remaining
│                                manual steps (launcher-package-specific, reference
│                                SkyrimNetMultiProxy.dll/.ini by name)
├── START HERE.txt             — plain-language quick-start for the zip
└── LICENSE-proxy.txt         — MIT (galanx + rhinos0608 notices travel with the code)
```

There is intentionally **no** `proxy.ini.example`: the current runtime reads no `proxy.ini`
(verified — zero references in `proxy.py` / `proxy_internal/`). The old `proxy.ini`-based
`AutoCloseWithSkyrim` / `EnableLogging` / `SkyrimProcess` settings below are historical — they
describe the previous single-file proxy generation and do not apply to the packaged runtime
until it actually supports them.

## Staging and building a release — the dev/public boundary

`RELEASING.md` is the authority on the full release process and the exact manifest; this section
covers the two scripts that enforce it.

`release-staging\` (gitignored) is the one place a sensitive file can leak into a published
zip — `config.json` (live API keys) sits one character away from `config.example.json` (which does
ship) in this repo's own root, so a folder copy or a glob is a real credential-leak risk, not a
hypothetical one. Both scripts below treat that as the thing to guard against, not just to document.

**`sync-release-staging.ps1`** stages the release files from this same tree. It reads every `.py`
path from `proxy-runtime.manifest` plus the four named non-code files (`requirements.txt`,
`config.example.json`, `start-proxy.bat`, `LICENSE-proxy.txt` renamed on the way in for the license).
Never a folder copy, never a wildcard — a new file added to the repo root does not get staged until
someone deliberately adds it to the manifest. Before copying anything, it checks `release-staging\`
itself against RELEASING.md's "What must never cross" list and refuses (naming the offender) if
anything on it is already sitting there — it does not delete the file itself, since a stray
`config.json` means someone copied the wrong thing and needs to know, not have it silently cleaned
up. Each staged file is hash-compared against what was already there so the output reports changed
vs. unchanged per file. After staging it asserts the staged `.py` set matches the manifest exactly —
missing or unexpected files fail the run.

**`build-release.ps1`** is the safety net, not the only one — it re-checks independently rather than
trusting the sync script ran, in case `release-staging\` was populated by hand (still supported; see
its own header comment). Before assembling the zip it: (1) re-runs the same never-publish check
against `release-staging\`, (1b) asserts every staged `.py` is in `proxy-runtime.manifest` and every
manifest entry is staged, (2) parses `config.example.json` and flags any non-empty value that
doesn't match an "obviously a placeholder" shape (empty string, `YOUR_...`, `CHANGE_ME`, `<...>`,
etc.) — shape-based rather than matching one vendor's key format, so it still catches a real key
staged under the example's name even for a provider added after this check was written, and (3)
re-checks the fully assembled staging tree right before `Compress-Archive` runs, since that's the
literal input to the zip. After the zip exists, it opens it back up and asserts the RELEASING.md
step 5 positives — exactly one top-level `SkyrimNet MultiProxy\` folder, the full runtime tree per
`proxy-runtime.manifest`, `LICENSE-proxy.txt` present, no `config.json`/`proxy.ini`/`.log`/`tests`/
`prompts` anywhere inside — before printing
success, so a broken assumption fails the build instead of shipping quietly.

The never-publish name/dir list is duplicated between the two scripts rather than factored into a
shared module (matching this repo's existing convention of small, independent scripts); RELEASING.md
is the authority both were written against, so if that list changes there, both scripts need the
matching edit.

## `apply-skyrim-watcher.py` — maintenance tool, not part of normal use

`Claude-SkyrimNet-Proxy`'s `proxy.py` needs a few fixes/additions to work well launched this way.
**These are already applied to the `proxy.py` bundled in every release** — a normal user never
needs to run this script.

It exists so the exact same changes can be reproduced against a *different* or future copy of the
upstream file (a fresh `git clone` of `galanx/Claude-SkyrimNet-Proxy`, or a later upstream
version) — e.g. if upstream releases updates and the patch script needs revalidation against them.
Validated 2026-07-26: running it against a completely fresh upstream clone reproduces the bundled
`proxy.py` (functionally identical; only differences are comments/import-order/one now-unused
parameter).

```cmd
python apply-skyrim-watcher.py path\to\proxy.py
python apply-skyrim-watcher.py path\to\proxy.py --enable   # also turns on auto-close immediately
```

- Takes a `.bak` backup before modifying anything.
- Safe to run more than once — detects if a patch is already applied.
- Safe to run against a version where the upstream PR has already merged — bug-fix hunks are
  skipped automatically rather than double-applied or conflicting.
- Creates `proxy.ini` alongside `proxy.py` if it doesn't exist yet. (Historical — applies only to
  the previous single-file proxy generation, not the current `proxy_internal/` runtime, which
  reads no `proxy.ini`.)

### Bug fixes (two also submitted as [PR #5](https://github.com/galanx/Claude-SkyrimNet-Proxy/pull/5) upstream)

Applied unconditionally, and auto-skipped once upstream has merged/already has them:

- **`UnboundLocalError` on every streaming response.** `_usage` was only assigned in the
  non-streaming branch of `call_api_direct` but referenced unconditionally after the `if/else`.
  Since Claude always returns `text/event-stream`, the non-streaming branch never ran — every
  request through that path returned HTTP 500. *(In PR #5.)*
- **Token counts missing from `call_api_direct` log lines.** The streaming branch only parsed
  `content_block_delta` events and never collected usage data. Fixed by adding
  `message_start`/`message_delta` parsing (matching what `call_api_streaming_with_retry` already
  did), so log lines now show `| in=N out=N tok` consistently. *(In PR #5.)*
- **Auth-capture condition too strict.** Upstream requires both `"system"` and `"messages"` keys
  in the parsed request body before capturing auth — real requests with no system prompt never
  triggered a capture. Broadened to require only `"messages"`.
- **Auth header fallback.** The captured-auth log line now also checks `x-api-key`, not just
  `Authorization`, since some captures use that header instead.
- **OpenRouter key handling crash.** Upstream extracted the key per-request via an unguarded
  `request.headers.get("authorization").removeprefix(...)`, raising an unhandled `AttributeError`
  (an ugly 500) if the header was simply absent. Replaced with a module-level `openrouter_api_key`
  loaded once from `config.json`, plus a proper "not configured" error instead of a crash.
- **Default model bump.** `DEFAULT_MODEL` updated to a more current model — a preference, not a
  bug fix, and one that will go stale again; check `_hunk_default_model` if it needs updating.

### Skyrim-specific additions (`proxy.ini`) — historical, previous generation only

The settings below belong to the previous single-file proxy generation and the `proxy.ini` that
`apply-skyrim-watcher.py` manages. They do **not** apply to the current packaged runtime
(`proxy.py` + `proxy_internal/`), which reads no `proxy.ini` — kept here so the history of what
the patch script does is still documented in one place.

```ini
[General]
; Shut the proxy down when Skyrim exits (polls every 10 s)
AutoCloseWithSkyrim = false

; Write proxy.log alongside proxy.py for debugging
EnableLogging = false

; Comma-separated Skyrim process names to watch
SkyrimProcess = SkyrimSE.exe, SkyrimVR.exe
```

Changes take effect the next time the proxy starts — no recompile or reinstall needed.

### Console title fix

When the proxy is launched by an external process (this plugin, or anything else), `claude
--print` runs at startup to capture auth headers and changes the console window's title as a side
effect. The patch restores the title to **Claude SkyrimNet Proxy** immediately after auth capture
completes -- this is the exact string `_hunk_console_title` inserts into a fresh upstream clone, so
it's left as-is here even after the 2026-09-13 SkyrimNet MultiProxy rename; the shipped plugin's
own console/error text uses the current name (see the sections above).

## Project structure

```
skyrimnet-multiproxy/
├── CMakeLists.txt          — build config; FetchContent handles all deps
├── apply-skyrim-watcher.py — maintenance-only patch script (see above)
├── SkyrimNetMultiProxy.ini — SKSE plugin config template (bundled in releases)
└── src/
    ├── PCH.h               — precompiled header (CommonLibSSE-NG)
    ├── main.cpp            — SKSE plugin entry point + logging
    ├── proxy_launcher.h    — launch result enum
    └── proxy_launcher.cpp  — pure Win32 process launcher (no CommonLibSSE deps)
```

## Dependencies (auto-downloaded by CMake)

- [CommonLibSSE-NG](https://github.com/CharmedBaryon/CommonLibSSE-NG) v3.7.0
- [fmt](https://github.com/fmtlib/fmt) 10.2.1
- [spdlog](https://github.com/gabime/spdlog) 1.13.0
- [rapidcsv](https://github.com/d99kris/rapidcsv) v8.83
