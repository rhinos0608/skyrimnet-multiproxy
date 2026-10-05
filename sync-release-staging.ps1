# Stages the release files from THIS repo's own tree into release-staging\
# (gitignored) -- no private -dev repo involved.
#
# Why this exists (read RELEASING.md first): the staging step is the one place
# a sensitive file can leak into a published zip. release-staging\ holds
# config.example.json -- one glob or one tab-completion slip away from staging
# the REAL config.json instead, which carries live API keys in plaintext.
# This script never globs and never copies a folder: every file below is named
# explicitly, and every .py must appear in proxy-runtime.manifest.
#
# Usage:
#   .\sync-release-staging.ps1
#
# Requires: nothing but this checkout. Runtime source (proxy.py +
# proxy_internal\) is tracked here; this script only stages it.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$stagingDir = Join-Path $root "release-staging"

# --- Refuse to proceed if a never-publish file is already sitting in release-staging\ ---
# Mirrors RELEASING.md's own "What must never cross" table -- keep this list in step by hand if
# that table changes; RELEASING.md is the authority, this is just its enforcement.
$neverPublishNames = @("config.json", "proxy.ini", "pytest.ini", "requirements-dev.txt", "CLAUDE.md", "TECHNICAL.md", "README.md", ".env")
$neverPublishDirs = @("prompts", "tests", "design", "docs", "__pycache__", ".git")

if (Test-Path $stagingDir -PathType Container) {
    $offenders = @()
    foreach ($name in $neverPublishNames) {
        $p = Join-Path $stagingDir $name
        if (Test-Path $p -PathType Leaf) { $offenders += $p }
    }
    foreach ($name in $neverPublishDirs) {
        $p = Join-Path $stagingDir $name
        if (Test-Path $p) { $offenders += $p }
    }
    $logFiles = Get-ChildItem $stagingDir -Filter "*.log" -File -ErrorAction SilentlyContinue
    if ($logFiles) { $offenders += $logFiles.FullName }

    if ($offenders.Count -gt 0) {
        Write-Host ""
        Write-Host "REFUSING TO PROCEED -- release-staging\ already contains a file that must never" -ForegroundColor Red
        Write-Host "be published (see RELEASING.md's own 'What must never cross' table):" -ForegroundColor Red
        foreach ($o in $offenders) { Write-Host "  - $o" -ForegroundColor Red }
        Write-Host ""
        Write-Host "Not deleted automatically -- someone copied the wrong thing and needs to know." -ForegroundColor Red
        throw "Aborting: forbidden file(s) present in release-staging\."
    }
}

New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null

# --- The manifest: proxy-runtime.manifest (all .py) + the four non-code files. ---
# .py entries come from proxy-runtime.manifest -- the explicit allowlist, one path per line,
# comments/blank lines ignored. Non-code files are named explicitly below, never globbed.
$manifestPath = Join-Path $root "proxy-runtime.manifest"
if (-not (Test-Path $manifestPath -PathType Leaf)) {
    throw "proxy-runtime.manifest is missing at $manifestPath -- it names every .py the release may contain."
}
$pyFiles = Get-Content $manifestPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" -and -not $_.StartsWith("#") }

$staticFiles = @("requirements.txt", "config.example.json", "start-proxy.bat", "LICENSE-proxy.txt")

Write-Host "Staging from: $root (this checkout -- no dev repo)"
Write-Host "Staging into: $stagingDir"
Write-Host ""

$results = @()
foreach ($rel in ($pyFiles + $staticFiles)) {
    $srcPath = Join-Path $root $rel
    $dstPath = Join-Path $stagingDir $rel

    if (-not (Test-Path $srcPath -PathType Leaf)) {
        throw "Manifest source missing: $srcPath -- has it moved? Update proxy-runtime.manifest and this script together."
    }

    $dstParent = Split-Path $dstPath -Parent
    New-Item -ItemType Directory -Path $dstParent -Force | Out-Null

    # "Changed since the last release" -- compared before overwriting, since Copy-Item -Force below
    # destroys the previous staged copy this comparison needs.
    $changed = $true
    if (Test-Path $dstPath -PathType Leaf) {
        $oldHash = (Get-FileHash $dstPath -Algorithm SHA256).Hash
        $newHash = (Get-FileHash $srcPath -Algorithm SHA256).Hash
        $changed = $oldHash -ne $newHash
    }

    Copy-Item -Path $srcPath -Destination $dstPath -Force

    $results += [pscustomobject]@{
        File    = $rel
        Changed = if ($changed) { "changed" } else { "unchanged" }
    }
}

# --- Fail closed: no staged .py may exist outside the manifest, and no sensitive
# filename may have slipped in (e.g. a real config.json copied by hand). ---
$stagedPy = Get-ChildItem $stagingDir -Recurse -Filter "*.py" -File -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName.Substring($stagingDir.Length + 1) -replace '\\', '/' }
$unexpected = $stagedPy | Where-Object { $pyFiles -notcontains $_ }
if ($unexpected) {
    foreach ($u in $unexpected) { Write-Host "  UNEXPECTED STAGED .py: $u" -ForegroundColor Red }
    throw "Aborting: staged .py files outside proxy-runtime.manifest."
}
$missing = $pyFiles | Where-Object { -not (Test-Path (Join-Path $stagingDir $_) -PathType Leaf) }
if ($missing) {
    foreach ($m in $missing) { Write-Host "  MISSING STAGED FILE: $m" -ForegroundColor Red }
    throw "Aborting: manifest files missing from release-staging\."
}

Write-Host "Staged:"
$results | ForEach-Object {
    Write-Host ("  {0,-48} {1}" -f $_.File, $_.Changed)
}
Write-Host ""
Write-Host "Manifest check passed: every staged .py is allowlisted, nothing missing." -ForegroundColor Green
Write-Host "Done. Next: .\build-release.ps1"
