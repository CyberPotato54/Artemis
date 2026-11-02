#Requires -Version 7.2
# Scans for secrets with gitleaks, the same pinned version the forge's secret-scan job runs (8.30.1), so a leak is
# caught before it reaches the forge.
#
#   -Mode tree      (default) the working tree: the tracked files and the untracked files git does not ignore,
#                   copied to a temp folder and scanned there, so target/ and other ignored folders cost nothing.
#   -Mode unpushed  the commits on no remote branch (the whole history if there is no remote).
#   -Mode range     the commits in -Range (anything `git log` takes, such as a..b or "<sha> --not --remotes").
#   -Mode dir       the files under -Path, as they are (used for the public export by check-public-snapshot.ps1).
#
# The tree, unpushed and range modes read the repository the current folder is in. They honour .gitleaksignore
# (fingerprints) and `gitleaks:allow` comments. The dir mode reads only the .gitleaksignore inside -Path, if any, so
# the private repository's fingerprints never hide something in an export.
# Every mode uses --redact: a finding is reported without the secret. Exit codes: 0 clean, 1 a finding (the report is
# printed), 3 the scan could not run (a bad range, a failed download, a hash mismatch; the reason is on stderr after
# "scan-secrets: "). A clean result is never printed for a scan that did not run.
# The rules are gitleaks' defaults, passed explicitly from tools/gitleaks.toml, and GITLEAKS_* environment variables
# are cleared for the run, so a .gitleaks.toml in the scanned content cannot replace them.
#
# The gitleaks binary is gitleaks_8.30.1_windows_x64.zip from this repository's forge release ci-tools/gitleaks-v8.30.1,
# downloaded once with the credential git already holds for the forge, checked against a pinned SHA-256 and cached in
# %LOCALAPPDATA%\artemis-tools\gitleaks-8.30.1 (the zip stays there and is checked on every run). A gitleaks on PATH is
# never used, and nothing is fetched from GitHub. The download URL prefix (the release's asset folder, without the
# file name) is the git config key artemis.gitleaksUrl, which tools/install-hooks.ps1 sets; -ForgeUrl <prefix>
# overrides it for one run. With neither, the script stops and names the key.
# ARTEMIS_TOOLS_DIR replaces %LOCALAPPDATA%\artemis-tools (the tests use it).
param(
  [ValidateSet('tree', 'unpushed', 'range', 'dir')]
  [string]$Mode = 'tree',
  [string]$Range,
  [string]$Path,
  [string]$ForgeUrl
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$version = '8.30.1'
$archive = "gitleaks_${version}_windows_x64.zip"
$pinned = 'd29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e'
$mismatch = 'gitleaks download does not match its pinned SHA-256'

function Get-Sha256([string]$file) {
  (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
}

# scheme://host[:port] of a URL, without any credentials, path or query.
function Get-ForgeBase([string]$url) {
  $uri = $null
  if (-not [uri]::TryCreate($url, [System.UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -notin @('http', 'https') -or -not $uri.Host) {
    return $null
  }
  $base = "$($uri.Scheme)://$($uri.Host)"
  if (-not $uri.IsDefaultPort) { $base += ":$($uri.Port)" }
  $base
}

# The credential git holds for this forge, as an Authorization header value. It is never printed and never put on a
# command line: it comes from `git credential fill` on stdin and goes out in a header object.
function Get-AuthHeader([string]$base) {
  $uri = [uri]$base
  $hostPort = if ($uri.IsDefaultPort) { $uri.Host } else { "$($uri.Host):$($uri.Port)" }
  $env:GIT_TERMINAL_PROMPT = '0'
  $reply = "protocol=$($uri.Scheme)`nhost=$hostPort`n`n" | git credential fill 2>$null
  if ($LASTEXITCODE -or -not $reply) { throw "git has no credential for $hostPort; sign in to the forge with git first" }
  $user = $null
  $pass = $null
  foreach ($line in $reply) {
    if ($line -like 'username=*') { $user = $line.Substring(9) }
    elseif ($line -like 'password=*') { $pass = $line.Substring(9) }
  }
  if (-not $pass) { throw "git has no credential for $hostPort; sign in to the forge with git first" }
  $raw = [System.Text.Encoding]::UTF8.GetBytes("${user}:${pass}")
  'Basic ' + [Convert]::ToBase64String($raw)
}

function Get-Gitleaks {
  $toolsRoot = if ($env:ARTEMIS_TOOLS_DIR) { $env:ARTEMIS_TOOLS_DIR } else { Join-Path $env:LOCALAPPDATA 'artemis-tools' }
  $cache = Join-Path $toolsRoot "gitleaks-$version"
  $zip = Join-Path $cache $archive
  $exe = Join-Path $cache 'gitleaks.exe'
  New-Item -ItemType Directory -Force $cache | Out-Null

  if (-not (Test-Path -LiteralPath $zip)) {
    $prefix = $ForgeUrl
    if (-not $prefix) {
      $prefix = [string](git -C (Split-Path $PSScriptRoot -Parent) config --get artemis.gitleaksUrl 2>$null)
    }
    if (-not $prefix) {
      throw 'no download URL for gitleaks: set the git config key artemis.gitleaksUrl (pwsh tools/install-hooks.ps1 sets it from origin) or pass -ForgeUrl <release asset URL prefix>'
    }
    $base = Get-ForgeBase $prefix
    if (-not $base) { throw "the gitleaks download URL is not an http(s) URL: $prefix" }
    if ($base -match '^https?://([^/:]+\.)?github(usercontent)?\.com(:|$)') {
      throw "the gitleaks download URL names GitHub ($base); it must be the forge release"
    }
    $url = $prefix.TrimEnd('/') + "/$archive"
    $part = "$zip.part"
    Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
    Write-Host "downloading gitleaks $version from the forge release"
    $headers = @{ Authorization = (Get-AuthHeader $base) }
    try {
      Invoke-WebRequest -Uri $url -Headers $headers -OutFile $part -UseBasicParsing
    } catch {
      Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
      throw "gitleaks download failed ($($_.Exception.Message))"
    }
    if ((Get-Sha256 $part) -ne $pinned) {
      Remove-Item -LiteralPath $part -Force
      throw $mismatch
    }
    Move-Item -LiteralPath $part -Destination $zip -Force
  } elseif ((Get-Sha256 $zip) -ne $pinned) {
    Remove-Item -LiteralPath $zip, $exe -Force -ErrorAction SilentlyContinue
    throw $mismatch
  }

  # The zip is verified. Keep gitleaks.exe equal to the entry in it, so a tampered or half-written exe is replaced.
  $za = [System.IO.Compression.ZipFile]::OpenRead($zip)
  try {
    $entry = $za.Entries | Where-Object { $_.FullName -eq 'gitleaks.exe' } | Select-Object -First 1
    if (-not $entry) { throw 'gitleaks.exe is not in the pinned zip' }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = $entry.Open()
    try { $want = [System.BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
    if (-not (Test-Path -LiteralPath $exe) -or (Get-Sha256 $exe) -ne $want) {
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $exe, $true)
    }
  } finally {
    $za.Dispose()
  }
  $exe
}

# Runs gitleaks with $gitleaksArgs. Leaks are reported with exit code 2, so a gitleaks failure (which exits 1) is told
# apart. Returns 0 when clean or 1 after printing the report on a finding; throws on anything else.
# $inFolder, when given, is the current folder for the run, so a scan of "." reports paths (and .gitleaksignore
# fingerprints) relative to the scanned folder.
function Invoke-Gitleaks([string[]]$gitleaksArgs, [string]$inFolder) {
  $config = Join-Path $PSScriptRoot 'gitleaks.toml'
  if (-not (Test-Path -LiteralPath $config)) { throw "no rule file at $config" }
  if ($inFolder) { Push-Location -LiteralPath $inFolder }
  # GITLEAKS_* variables (GITLEAKS_CONFIG, GITLEAKS_CONFIG_TOML, ...) can replace the rules: clear them for the run.
  $saved = @{}
  foreach ($v in Get-ChildItem Env: | Where-Object { $_.Name -like 'GITLEAKS_*' }) {
    $saved[$v.Name] = $v.Value
    Remove-Item -LiteralPath "Env:$($v.Name)"
  }
  try {
    $out = & $script:exe @gitleaksArgs --config $config --redact --verbose --no-banner --no-color --exit-code 2 2>&1
    $code = $LASTEXITCODE
  } finally {
    if ($inFolder) { Pop-Location }
    foreach ($k in $saved.Keys) { Set-Item -LiteralPath "Env:$k" -Value $saved[$k] }
  }
  # `gitleaks git` exits 0 after logging an ERR line when its own git log fails (a bad range, say); that is no scan.
  $errors = @($out | ForEach-Object { "$_" } | Where-Object { $_ -match '^\S+\s+ERR\s' })
  if ($errors.Count) {
    $errors | ForEach-Object { [Console]::Error.WriteLine($_) }
    throw "gitleaks reported an error ($($errors[0]))"
  }
  if ($code -eq 0) {
    Write-Host "gitleaks ${version}: no leaks found"
    return 0
  }
  if ($code -eq 2) {
    $out | ForEach-Object { Write-Host "$_" }
    return 1
  }
  $out | ForEach-Object { [Console]::Error.WriteLine("$_") }
  throw "gitleaks failed with exit code $code"
}

function Invoke-Main {
  $script:exe = Get-Gitleaks

  if ($Mode -eq 'dir') {
    if (-not $Path) { throw '-Mode dir needs -Path <folder>' }
    $target = (Resolve-Path -LiteralPath $Path).Path
    return (Invoke-Gitleaks @('dir', '.', '--gitleaks-ignore-path', '.') $target)
  }

  $repo = (git rev-parse --show-toplevel 2>$null)
  if ($LASTEXITCODE -or -not $repo) { throw "$Mode mode runs inside a git repository" }
  $repo = (Resolve-Path -LiteralPath ([string]$repo)).Path

  switch ($Mode) {
    'tree' {
      $work = Join-Path ([System.IO.Path]::GetTempPath()) ('artemis-secrets-' + [guid]::NewGuid().ToString('N'))
      New-Item -ItemType Directory $work | Out-Null
      try {
        $prev = [Console]::OutputEncoding
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
        try { $listed = (git -C $repo ls-files -z --cached --others --exclude-standard) -join '' }
        finally { [Console]::OutputEncoding = $prev }
        if ($LASTEXITCODE) { throw 'git ls-files failed' }
        foreach ($rel in ($listed -split "`0" | Where-Object { $_ })) {
          $src = Join-Path $repo $rel
          if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { continue }
          $dst = Join-Path $work $rel
          New-Item -ItemType Directory -Force (Split-Path $dst -Parent) | Out-Null
          Copy-Item -LiteralPath $src -Destination $dst
        }
        return (Invoke-Gitleaks @('dir', '.', '--gitleaks-ignore-path', '.') $work)
      } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
    'unpushed' { $logOpts = 'HEAD --branches --not --remotes' }
    'range' {
      if (-not $Range) { throw '-Mode range needs -Range <a..b>' }
      $logOpts = $Range
    }
  }
  # The range must be one git can walk, and only revisions and the few ref selectors may appear in it.
  $revArgs = @($logOpts -split '\s+' | Where-Object { $_ })
  foreach ($a in $revArgs) {
    if ($a.StartsWith('-') -and $a -notin @('--not', '--remotes', '--branches', '--tags', '--all')) {
      throw "the range holds an option that is not allowed here: $a"
    }
  }
  git -C $repo rev-list --quiet @revArgs 2>$null
  if ($LASTEXITCODE) { throw "git cannot walk the range '$logOpts'" }
  return (Invoke-Gitleaks @('git', $repo, "--log-opts=$logOpts", '--gitleaks-ignore-path', $repo))
}

try {
  exit ([int]@(Invoke-Main)[-1])
} catch {
  [Console]::Error.WriteLine("scan-secrets: $($_.Exception.Message)")
  exit 3
}
