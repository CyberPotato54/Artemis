#Requires -Version 7.2
# Turns on the repository's git hooks and tells the secret scan where to download gitleaks from. It is idempotent.
#  - artemis.gitleaksUrl = the URL prefix of the forge release that holds gitleaks_8.30.1_windows_x64.zip (the zip's
#    name is added by tools/scan-secrets.ps1). -GitleaksUrl gives it. Without the parameter, a value already set is
#    kept, and otherwise it is derived once from this clone's origin: <origin without .git>/releases/download/<tag>.
#    A GitHub address is refused: the scan never downloads from GitHub.
#  - core.hooksPath = tools/git-hooks (the pre-push secret scan). The setting lives in the shared repository config, so
#    it covers every worktree of this clone, and each worktree runs the hooks from its own checkout; a checkout that
#    has no tools/git-hooks runs no hook. It is set only once the scan can get gitleaks: a download URL is set, or the
#    pinned zip is already in the tools folder (ARTEMIS_TOOLS_DIR or %LOCALAPPDATA%\artemis-tools). Otherwise every push
#    would fail at the hook (a clone of the public GitHub repository has no forge release to download from), so the
#    hook stays off and the script says what the scan needs. The public repository's secret check is the `secrets` job
#    in .github/workflows/ci.yml.
param(
  [string]$GitleaksUrl
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$tag = 'ci-tools%2Fgitleaks-v8.30.1'
$githubUrl = '^https?://([^/:]+\.)?github(usercontent)?\.com(:|/|$)'

# Whether the pinned gitleaks zip is already cached and matches the SHA-256 tools/scan-secrets.ps1 pins.
function Test-CachedZip {
  $text = [System.IO.File]::ReadAllText((Join-Path $PSScriptRoot 'scan-secrets.ps1'))
  $m = [regex]::Match($text, "\`$version = '([0-9.]+)'")
  $p = [regex]::Match($text, "\`$pinned = '([0-9a-f]{64})'")
  if (-not $m.Success -or -not $p.Success) { return $false }
  $toolsRoot = if ($env:ARTEMIS_TOOLS_DIR) { $env:ARTEMIS_TOOLS_DIR } else { Join-Path $env:LOCALAPPDATA 'artemis-tools' }
  $zip = Join-Path $toolsRoot "gitleaks-$($m.Groups[1].Value)\gitleaks_$($m.Groups[1].Value)_windows_x64.zip"
  if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) { return $false }
  return ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -eq $p.Groups[1].Value)
}

# The download URL: the parameter, else the value already set, else one derived from origin; $null when there is none
# (and a note says why). An explicit address that names GitHub is an error; a derived or stored one is only unusable.
$url = $null
$note = $null
if ($GitleaksUrl) {
  if ($GitleaksUrl -match $githubUrl) { throw 'the gitleaks download URL must name the forge release, and this address is GitHub' }
  $url = $GitleaksUrl
} else {
  $current = [string](git -C $repo config --get artemis.gitleaksUrl 2>$null)
  if ($current -and $current -notmatch $githubUrl) {
    $url = $current
    "artemis.gitleaksUrl = $current (kept)"
  } else {
    $origin = [string](git -C $repo remote get-url origin 2>$null)
    $uri = $null
    if (-not [uri]::TryCreate($origin, [System.UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -notin @('http', 'https')) {
      $note = 'this clone has no http(s) origin to derive the gitleaks download URL from (run again with -GitleaksUrl <release asset URL prefix>)'
    } elseif ($uri.Host -match '(^|\.)github(usercontent)?\.com$') {
      $note = "this clone's origin is GitHub, which holds no gitleaks release for the scan"
    } else {
      $path = $uri.AbsolutePath.TrimEnd('/')
      if ($path.EndsWith('.git')) { $path = $path.Substring(0, $path.Length - 4) }
      $hostPort = if ($uri.IsDefaultPort) { $uri.Host } else { "$($uri.Host):$($uri.Port)" }
      $url = "$($uri.Scheme)://$hostPort$path/releases/download/$tag"
    }
  }
}

if ($url -and $url -ne [string](git -C $repo config --get artemis.gitleaksUrl 2>$null)) {
  git -C $repo config artemis.gitleaksUrl $url
  if ($LASTEXITCODE) { throw 'git config failed' }
  "artemis.gitleaksUrl = $url"
}

if ($url -or (Test-CachedZip)) {
  git -C $repo config core.hooksPath tools/git-hooks
  if ($LASTEXITCODE) { throw 'git config failed' }
  "core.hooksPath = $(git -C $repo config --get core.hooksPath)"
} else {
  if (-not $note) { $note = 'no gitleaks download URL is set' }
  Write-Warning "the pre-push hook stays off: $note, and the pinned gitleaks zip is not cached. The local scan needs the forge release (-GitleaksUrl) or a pre-seeded ARTEMIS_TOOLS_DIR holding gitleaks-8.30.1\gitleaks_8.30.1_windows_x64.zip. The public repository's secret check is the secrets job in .github/workflows/ci.yml."
}
