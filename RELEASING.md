# Releasing SkyrimNet MultiProxy

Two parts build one release, and they must not bleed into each other. This doc is the authority on
which one owns what, exactly which files cross the line, and which files must never cross it.

- **This repo's own Python tree** — the proxy itself: `proxy.py`, `proxy_internal/`, their own
  `requirements.txt`, templates, and the working notes. This checkout is the Python source of
  truth; there is no separate private `-dev` repo anymore. Do not reintroduce one.
- **This repo's SKSE plugin** — what the world sees alongside the proxy: the plugin (`src/`, the
  built `.dll`), the release zip, the public `README.md` / `TECHNICAL.md` / `RELEASE_NOTES.md`,
  and the GitHub releases.

The release zip is assembled from this same tree through `release-staging\` (gitignored), which
is why a hand-copy step has always existed here — and why that step is the one place a private
file can leak into a published zip.

> **Status: UNRELEASED.** The `proxy.py` + `proxy_internal/` tree in this checkout has not shipped
> in any release yet. Nothing in this doc claims otherwise — do not describe the packaged
> runtime, the manifest, or any new provider support as shipped until a release actually goes out.

---

## What goes into the zip

The Python runtime plus four non-code files, into `release-staging\`. Nothing else, ever.
Every `.py` is named in `proxy-runtime.manifest` (the explicit allowlist — `proxy.py` plus the
full `proxy_internal/` tree, 61 files). The four non-code files are named explicitly in
`sync-release-staging.ps1`. No folder copies, no wildcards, ever.

| Staged file | Comes from | Notes |
| :--- | :--- | :--- |
| `proxy.py` + `proxy_internal/` tree | this repo, per `proxy-runtime.manifest` | The only files that change in most releases. `proxy.py` alone is NOT enough — the runtime is a package now and crashes on its first `from proxy_internal...` import without the tree |
| `requirements.txt` | this repo root | Runtime deps only — dev-only packages (`pytest`, `pytest-asyncio`, `httpx`, `google-genai`) are stripped and never ship |
| `config.example.json` | this repo root | The template, with placeholder keys |
| `start-proxy.bat` | this repo root | Cd's to its own folder (`%~dp0`) — no hardcoded paths |
| `LICENSE-proxy.txt` | this repo's proxy license | Renamed copy. MIT requires galanx's and rhinos0608's notices to travel with the bundled code |

There is intentionally **no** `proxy.ini.example`: the current runtime reads no `proxy.ini`
(verified — zero references in `proxy.py` / `proxy_internal/`). Do not invent one until the
runtime actually supports it.

## What must never cross

Each of these has a real reason, not a tidiness reason:

| Never publish | Why |
| :--- | :--- |
| **`config.json`** | Live API keys in plaintext. It sits one character away from `config.example.json` in the same folder, so a careless glob or a tab-completion slip publishes real credentials. This is the single highest-risk file in the tree. |
| `proxy.ini`, `proxy.log` | A live `proxy.ini` (personal paths) never ships; `proxy.log` holds request history from real play sessions. No `proxy.ini.example` ships either — the runtime reads no `proxy.ini`. |
| `prompts/` | Queue, handoffs, board state — the working record, not a product. |
| `.env`, auth stores, `tests/`, `pytest.ini` | Development-only and secrets-adjacent. A user running the release never needs them. |
| `design/`, `docs/`, `CLAUDE.md` | Internal material, never published. |
This repo's `TECHNICAL.md` and `README.md` are the public, player-facing versions — never overwritten
from anywhere else. They have no private counterparts anymore.

**The rule that catches all of it:** stage by the explicit manifest (`proxy-runtime.manifest` +
four named non-code files), never by copying a folder or a wildcard. `build-release.ps1` refuses
to build if a forbidden file is sitting in `release-staging\`.

---

## Releasing

1. **Confirm what actually changed.** Read the commits since the last release tag.
   If only the runtime tree changed, this is a proxy-only release — the SKSE plugin's own **code** doesn't
   need touching, but see the next step: its **version** still needs bumping and rebuilding, every
   time, with no exception.
2. **The plugin's version ALWAYS moves to match the release, every release — no exceptions, even
   when `src/` didn't change at all.** This used to be optional ("knowingly ship the plugin at its
   existing version" was a valid choice, noted in the release notes if it might confuse anyone) —
   **it no longer is.** The dashboard (`proxy.py`, since the v3.0.0 version-display feature) now
   actively shows the proxy's version next to the plugin's own reported version and flags it as a
   visible warning if they disagree. A "proxy-only" release that skips the plugin's version bump
   doesn't just risk *looking* stale — it now makes every single player's dashboard show a false
   "these don't match, something might be wrong" warning that was never actually a real problem.
   Bumping-and-rebuilding a functionally-unchanged plugin, purely so its reported version stays
   current, is now a mandatory step of every release, not a judgment call.

   Two numbers must move together and must always agree with each other, every time:
   - `skse-project.json`'s `Version` — names the plugin and drives the release zip's own filename.
   - `CMakeLists.txt`'s `project(SkyrimNetMultiProxy VERSION X.Y.Z ...)` — the thing that actually
     feeds the *compiled* version string (`SKYRIMNET_MULTIPROXY_VERSION_STRING`, what
     `SkyrimNetMultiProxy.log` reports and what the new dashboard check reads). **These two
     genuinely drifted apart for real in the v3.0.0 release** — `skse-project.json` got bumped,
     `CMakeLists.txt` didn't, and the compiled plugin kept reporting the old version despite the zip
     being named correctly. Check both, every release, not just one.
   - After bumping `CMakeLists.txt`, a real reconfigure + rebuild is required (`project()`'s
     `VERSION` only takes effect after `cmake -B build -S ...` regenerates — `cmake --build` alone
     against a stale cache won't pick it up). Confirm the rebuilt `.dll` actually reports the new
     version (via `SkyrimNetMultiProxy.log` or the `tools/launch-proxy-test` harness) before moving
     on — don't trust the version number in a file alone.
3. **Stage the runtime** — `.\sync-release-staging.ps1`, which stages exactly the manifest above and
   refuses anything else.
4. **Build the zip** — `.\build-release.ps1`. It reads the version from `skse-project.json` and writes
   `github-releases\SkyrimNetMultiProxy-vX.Y.Z.zip`.
5. **Check the zip before it goes anywhere.** List its contents and confirm: the full runtime tree
   per `proxy-runtime.manifest` (`proxy.py` + `proxy_internal/`), the four non-code files, no
   `config.json`, no `proxy.ini`, no `.log`, no `tests`, no `prompts`, one top-level
   `SkyrimNet MultiProxy\` folder, and `LICENSE-proxy.txt` present. A published zip can't be
   unpublished from anyone who already has it.
6. **Write the notes.** `RELEASE_NOTES.md` gets the new version's section on top, and the GitHub
   release body is the same text. Design side drafts, Gemini gets a real pass at it, the director
   approves the final wording. The header sells **one** flagship feature, not a flat list.

   **House style — a release page should look like the README.** Same icon vocabulary, so everything
   published reads as one family:

   | Section | Header |
   | :--- | :--- |
   | What changed | `## ✨ What's New` |
   | Smaller fixes | `## 🔧 Improvements & Polish` |
   | Caveats, upgrade notes | `## 📋 Good to Know` |
   | Who to thank | `## 🤝 Special Thanks` |
   | Where to report things | `## 💬 Need Help?` |

   Two more rules that go with it:

   - **A banner image at the very top**, the same way `README.md` opens with one. It lives in
     `assets/` (past the `assets/*` ignore rule, with its own `!assets/…` exception), shrunk to
     1600px wide, with the full-res original kept beside it locally. The release body references it
     by absolute `raw.githubusercontent.com` URL — a release body can't resolve a repo-relative path.
   - **No `# Heading` inside the release body.** GitHub already prints the release title above it, so
     a body-level H1 shows the same line twice. `RELEASE_NOTES.md` keeps its `# vX.Y.Z — …` heading,
     since that's what separates one version from the next in a single file; strip it when publishing.
7. **Publish** — tag, `gh release create`, attach the zip. Never on your own initiative: the director
   approves the notes and says go.
8. **Afterwards — three things, in order, and don't skip the first one:**

   1. **Update the public `README.md`.** This step has actually been missed before (v3.0.0) —
      publishing the release does NOT do this automatically, and it's easy to feel "done" right
      after tagging/publishing when the real user-facing doc hasn't moved yet. Concretely: does
      `## ✨ Key Features` need a new bullet for this release's headline feature(s)? Does any
      existing bullet or tip now read stale (an old warning, an old workaround, an old provider
      count) given what just shipped? Write it in the README's **own voice** — don't just paste
      this repo's bullets or `RELEASE_NOTES.md` prose across; this is a living "what can it do right now"
      doc, not a changelog entry.
   2. **Write/update the Discord post** — see its own dedicated guidance below.
   3. **Clear the release's TODO lines.**

   **GitHub and Discord are two different kinds of document.** The director's own framing (2026-09-16):
   *"github is the series of announcements — Discord is what the current build is about."*

   | | GitHub releases | The Discord post |
   | :--- | :--- | :--- |
   | What it is | One entry per version, kept forever | **One living post, replaced each release** |
   | Written for | Someone asking "what changed in this version" | Someone asking "what is this mod" |
   | Carries | Only that version's changes | Links, features, setup, requirements, ToS, credits — every time |
   | Overlap with last time | None | Most of it, and that's correct |

   So a release's notes are a changelog and stop there, while the Discord post is a full standalone
   description whose changelog block near the top is the part that actually changes. Most of that post
   being identical to last time is right, not lazy — and it should never be written as a terse "here's
   what's new" announcement.

   **The Links block carries three entries, every time** (the third added by the director on the v2.4.1
   post, 2026-09-16 — *"let's make sure we keep that going forward"*):

   ```
   - GitHub + full guide: <https://github.com/awesmdiver/skyrimnet-multiproxy>
   - Download: <https://github.com/awesmdiver/skyrimnet-multiproxy/releases/latest>
   - Issues / feature requests: <https://github.com/awesmdiver/skyrimnet-multiproxy/issues>
   ```

   The angle brackets suppress Discord's link preview. They only work with **no space inside them** —
   `<url >` renders the brackets as literal text instead, which is easy to introduce when editing the
   post by hand and easy to miss.

   It's written for **4,000 characters** (the director has Discord Nitro), not the default 2,000 — two
   posts were needlessly compressed before anyone checked. Measure the final text anyway: a copy pass
   will sail past a hard limit even when told not to, which has happened twice.

---

## Keeping the public tree honest

- Anything user-facing that lands in the runtime tree is a **flag**, not a publish: the public README only
  describes what a released version can actually do. A feature sitting unreleased in this tree doesn't
  belong there yet.
- This repo's own `TECHNICAL.md` describes the plugin, the release mechanics, and the packaged
  runtime tree. Proxy patch internals live in `apply-skyrim-watcher.py`'s own doc comments.
- When the proxy gains a provider, the release notes and public README are the only public surfaces
  that need to change. If you find yourself copying anything else across, stop and re-read the table
  above.

> **Unofficial credential use — account risk.** Some providers in the runtime authenticate by
> reusing a third-party CLI login rather than an official API key: the Command Code CLI fallback
> (OAuth tokens captured from the Codex CLI) and OpenCode Go fall in this bucket. That workflow
> may violate the provider's Terms of Service — access can be throttled or revoked, and the
> account behind the login can be affected. Where you are entitled to it, prefer the provider's
> official API path instead (for Codex, the Command Code Provider API route on the dashboard) —
> a real API key under your own billing, no intercepted credentials. The dashboard and README
> carry the same warning; keep all three in step.
