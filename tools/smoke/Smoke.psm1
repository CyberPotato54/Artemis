#Requires -Version 7.2
# The client smoke: one real client (OpenCode or Claude Code) fixes a planted bug in a scratch
# repo against the real `artemis serve`. PowerShell owns the run (the root, the key, the server, the pass rule, the stop,
# the evidence); `tools/py/capture_clients.py smoke` owns the client (every guard, and the proof of where its requests
# went). The rules this module enforces:
#   - no run starts without the canary marker for these exact files (Assert-SmokeCanary, checked as capture_clients.py's
#     require_smoke_canary checks it, before anything is created);
#   - the run root is a direct child of T:\artemis-oracle\runs\m3c-smoke, the key lives under m3c-smoke\keys\ (outside
#     the root Assert-NoKey scans), and the server is the only thing on 127.0.0.1:8090;
#   - after a client has run in a scratch repo, the repo is not trusted: git is never run there, test_calc.py is restored
#     by a file copy, and the tests run under a cleared environment with `python -I -B`;
#   - the verdict: a guard's stop, a foreign credential, a server error or a found key fails the run;
#   - the key file is owner-only (New-ArtemisKey), and after a clean scan it is deleted and OpenCode's two configs have
#     their key blanked, so the run's key does not stay on disk.
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot '..\ps\ArtemisServer.psm1') -Force
. (Join-Path $PSScriptRoot '..\ps\Python.ps1')

$script:SmokeBase = 'T:\artemis-oracle\runs\m3c-smoke'
# The files the canary marker binds, by repo-relative path (capture_clients.py's SMOKE_BOUND).
$script:Bound = @('tools/py/capture_clients.py', 'tools/py/record_clients.py', 'tools/smoke/opencode.ps1',
  'tools/smoke/claude-code.ps1', 'tools/smoke/Smoke.psm1', 'tools/ps/ArtemisServer.psm1', 'tools/ps/Python.ps1')
$script:Task = 'Fix the bugs in calc.py so that the tests in test_calc.py pass, then run the tests with python -m unittest -v.'
$script:ScratchFiles = @('calc.py', 'test_calc.py', 'README.md', '.gitignore')
$script:ClientTimeoutS = 900
$script:TestsTimeoutS = 120
$script:ExpectedTests = 4

# --- private helpers ---------------------------------------------------------------------------------------------

function Resolve-FullPath([string]$Path) {
  return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Get-SmokeRepo { return (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) }
# The checkout this module is part of: what Invoke-Smoke hashes and copies the scratch repo from.
$script:RepoRoot = Get-SmokeRepo

function Test-SamePath([string]$A, [string]$B) {
  return ([System.IO.Path]::GetFullPath($A).TrimEnd('\') -ieq [System.IO.Path]::GetFullPath($B).TrimEnd('\'))
}

function Write-Utf8([string]$Path, [string]$Text) {
  [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

# Starts Exe and waits for it, TimeoutS at most (then the whole tree is killed). Stdout and stderr go to OutFile and
# ErrFile through Stream.CopyToAsync (two files, because two writers on one stream would interleave). With Env the child
# gets exactly that environment, nothing else. Returns @{exit; timed_out; seconds}. Never leaves the child running.
function Invoke-Bounded {
  param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string[]]$ArgList,
    [Parameter(Mandatory)][string]$OutFile,
    [Parameter(Mandatory)][string]$ErrFile,
    [Parameter(Mandatory)][int]$TimeoutS,
    [Parameter(Mandatory)][string]$WorkDir,
    [hashtable]$Env
  )
  $psi = [System.Diagnostics.ProcessStartInfo]::new($Exe)
  foreach ($a in $ArgList) { $psi.ArgumentList.Add($a) }
  $psi.WorkingDirectory = $WorkDir
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  if ($Env) {
    $psi.Environment.Clear()
    foreach ($k in $Env.Keys) { $psi.Environment[[string]$k] = [string]$Env[$k] }
  }
  $proc = [System.Diagnostics.Process]::new()
  $proc.StartInfo = $psi
  $outStream = [System.IO.FileStream]::new($OutFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
  $errStream = [System.IO.FileStream]::new($ErrFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
  $clock = [System.Diagnostics.Stopwatch]::StartNew()
  $timedOut = $false
  try {
    $null = $proc.Start()
    $proc.StandardInput.Close()
    $t1 = $proc.StandardOutput.BaseStream.CopyToAsync($outStream)
    $t2 = $proc.StandardError.BaseStream.CopyToAsync($errStream)
    if (-not $proc.WaitForExit($TimeoutS * 1000)) {
      $timedOut = $true
      try { $proc.Kill($true) } catch { }
      $null = $proc.WaitForExit(10000)
    }
    try { $null = [System.Threading.Tasks.Task]::WaitAll(@($t1, $t2), 10000) } catch { }
    $code = if ($proc.HasExited) { $proc.ExitCode } else { $null }
    return @{ exit = $code; timed_out = $timedOut; seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1) }
  } finally {
    # A stopped pipeline lands here with the child still running.
    try { if (-not $proc.HasExited) { $proc.Kill($true) } } catch { }
    $outStream.Dispose()
    $errStream.Dispose()
    $proc.Dispose()
  }
}

# SHA-256 (lower-case hex) of every bound file, in order. A bound file that is missing throws: the marker fails closed.
function Get-SmokeHashes([string]$Repo) {
  $out = [ordered]@{}
  foreach ($rel in $script:Bound) {
    $path = Join-Path $Repo ($rel -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      throw "$rel is missing: a smoke canary marker binds it, so no marker holds without it"
    }
    $out[$rel] = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData([System.IO.File]::ReadAllBytes($path))).ToLowerInvariant()
  }
  return $out
}

# The whole second of an instant, in UTC (the server's log time has second resolution).
function Get-WholeSecond([datetime]$When) {
  $utc = $When.ToUniversalTime()
  return [datetime]::new($utc.Ticks - ($utc.Ticks % [TimeSpan]::TicksPerSecond), [System.DateTimeKind]::Utc)
}

# A log line's `time` as a UTC instant. ConvertFrom-Json may have turned the string into a DateTime already.
function ConvertTo-LineTime($Value) {
  if ($Value -is [datetime]) { return $Value.ToUniversalTime() }
  return [System.DateTimeOffset]::Parse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture,
    [System.Globalization.DateTimeStyles]::AssumeUniversal -bor [System.Globalization.DateTimeStyles]::AdjustToUniversal).UtcDateTime
}

function Add-Names($Table, $Names) {
  foreach ($n in @($Names)) {
    if ($null -eq $n) { continue }
    $Table[[string]$n] = 1 + [int]($Table[[string]$n])
  }
}

function Copy-Scratch([string]$Repo, [string]$Dest) {
  $src = Join-Path $Repo 'tools\smoke\scratch'
  foreach ($f in $script:ScratchFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $src $f) -PathType Leaf)) { throw "the scratch repo is missing $f ($src)" }
  }
  New-Item -ItemType Directory -Force $Dest | Out-Null
  foreach ($f in $script:ScratchFiles) { Copy-Item -LiteralPath (Join-Path $src $f) -Destination (Join-Path $Dest $f) }
}

# --- the real parts the hooks default to --------------------------------------------------------------------------

# One `capture_clients.py smoke` run with a plain interpreter (the tools use only the standard library), as the runner calls
# its other Python tools. Returns @{exit; timed_out; result}: result is the file it wrote, or $null.
function Invoke-SmokeCapture {
  param($Root, $Client, $KeyFile, $ServerPid, $Project, $Out, $Task, $TimeoutS, $Repo, $Name)
  $python = Get-ArtemisPython
  $argList = @('-I', (Join-Path $Repo 'tools\py\capture_clients.py'),
    'smoke', '--client', $Client, '--root', $Root, '--key-file', $KeyFile, '--server-pid', "$ServerPid", '--task', $Task,
    '--timeout', "$TimeoutS", '--out', $Out, '--project', $Project)
  # The launcher's own bound is its timeout plus the discovery runs, the guards and the final sweep.
  $r = Invoke-Bounded -Exe $python -ArgList $argList -OutFile (Join-Path $Root "$Name.out.txt") -ErrFile (Join-Path $Root "$Name.err.txt") `
    -TimeoutS ($TimeoutS + 600) -WorkDir $Repo
  $result = $null
  if (Test-Path -LiteralPath $Out -PathType Leaf) {
    try { $result = Get-Content -LiteralPath $Out -Raw | ConvertFrom-Json } catch { $result = $null }
  }
  return @{ exit = $r.exit; timed_out = $r.timed_out; result = $result }
}

# `capture_clients.py blank-opencode-key`: sets apiKey to "" in OpenCode's two configs under Root. Throws unless it
# exits 0.
function Invoke-SmokeBlankKey {
  param($Root, $Repo)
  $python = Get-ArtemisPython
  $r = Invoke-Bounded -Exe $python -ArgList @('-I', (Join-Path $Repo 'tools\py\capture_clients.py'), 'blank-opencode-key', '--root', $Root) `
    -OutFile (Join-Path $Root 'blank-key.out.txt') -ErrFile (Join-Path $Root 'blank-key.err.txt') -TimeoutS 60 -WorkDir $Repo
  if ($r.timed_out -or $r.exit -ne 0) { throw "blank-opencode-key exited with $($r.exit) (timed out: $($r.timed_out)); see blank-key.err.txt in the run root" }
}

# The judge's own run of the tests, after a client has been in the repo. test_calc.py is put back from the repo's pristine
# copy by a file copy (no git: the client may have planted a .git/config), and the tests run as `python -I -B -m unittest
# discover -v -p test_calc.py` under a cleared environment, so the model's calc.py sees nothing of the user's. Passes only
# on exit 0, the four tests run and OK.
function Invoke-SmokeTests {
  param($Root, $Project, $Repo, $Name)
  $proj = Join-Path $Root $Project
  $pristine = Join-Path $Repo 'tools\smoke\scratch\test_calc.py'
  if ((Get-Item -LiteralPath $proj -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
    return @{ pass = $false; exit = $null; note = "$Project is a link, not a directory" }
  }
  $target = Join-Path $proj 'test_calc.py'
  if (Test-Path -LiteralPath $target) {
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer) { return @{ pass = $false; exit = $null; note = 'test_calc.py is a directory' } }
    # Whatever it is (a file, a symbolic link, a hard link) is deleted before the copy, so the copy cannot write through a
    # link. File.Delete removes the directory entry only: a link's target, or the other names of a hard-linked file, stay.
    # A delete or copy that fails (the client made the file read-only, say) is a failed attempt, not a failed run: the
    # pristine tests were not put back, so these tests are not run and the attempt does not pass. Attributes are not
    # changed (that could reach through a link), so catching is the whole handling.
    try {
      [System.IO.File]::Delete($target)
    } catch {
      return @{ pass = $false; exit = $null; note = "test_calc.py could not be removed to restore it, so the tests were not run: $($_.Exception.Message)" }
    }
  }
  try {
    [System.IO.File]::Copy($pristine, $target, $true)
  } catch {
    return @{ pass = $false; exit = $null; note = "test_calc.py could not be restored, so the tests were not run: $($_.Exception.Message)" }
  }
  $python = Get-ArtemisPython
  $home1 = Join-Path $Root 'home'
  $env1 = @{
    SystemRoot = $env:SystemRoot; WINDIR = $env:SystemRoot; PATH = $env:PATH; PATHEXT = $env:PATHEXT; COMSPEC = $env:COMSPEC
    USERPROFILE = $home1; HOME = $home1; TEMP = (Join-Path $home1 'Temp'); TMP = (Join-Path $home1 'Temp')
    # This runs model-written code unwatched: the discard proxy (nothing listens on port 9) makes any non-loopback request
    # that honours the proxy variables fail at once, as in the clients' own environment (capture_clients.py PROXY_VARS).
    HTTP_PROXY = 'http://127.0.0.1:9'; HTTPS_PROXY = 'http://127.0.0.1:9'; ALL_PROXY = 'http://127.0.0.1:9'
    NO_PROXY = '127.0.0.1,localhost'
  }
  $outFile = Join-Path $Root "$Name.out.txt"
  $errFile = Join-Path $Root "$Name.err.txt"
  # `discover` because -I keeps the working directory off sys.path, so a named module would not import; discovery puts the
  # start directory there itself. Only test_calc.py is looked for, so a test file the client added is not run.
  $r = Invoke-Bounded -Exe $python -ArgList @('-I', '-B', '-m', 'unittest', 'discover', '-v', '-s', '.', '-p', 'test_calc.py') -OutFile $outFile -ErrFile $errFile `
    -TimeoutS $script:TestsTimeoutS -WorkDir $proj -Env $env1
  $text = (Get-Content -LiteralPath $outFile -Raw) + (Get-Content -LiteralPath $errFile -Raw)
  $pass = ($r.exit -eq 0) -and -not $r.timed_out -and ($text -match "(?m)^Ran $($script:ExpectedTests) tests in ") -and ($text -match '(?m)^OK\s*$')
  $note = if ($r.timed_out) { "the tests timed out after $($script:TestsTimeoutS) s" } else { $null }
  return @{ pass = [bool]$pass; exit = $r.exit; note = $note }
}

# Refuses a -Config that is not a template the server could start from: the file must exist and fill with stand-in values
# (New-ArtemisConfig's rules, written to a temporary file that is removed). Run before the run creates anything.
function Assert-ConfigTemplate([string]$Config) {
  $Config = Resolve-FullPath $Config
  if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "the config template $Config does not exist" }
  $scratch = Join-Path ([System.IO.Path]::GetTempPath()) "smoke-config-check-$([guid]::NewGuid().ToString('N')).toml"
  try {
    $null = New-ArtemisConfig -Template $Config -Out $scratch -Model 'C:\check\model.gguf' -KeyFile 'C:\check\key.txt' -LogDir 'C:\check\logs'
  } catch {
    throw "the config template $Config does not fill: $($_.Exception.Message)"
  } finally {
    Remove-Item -LiteralPath $scratch -Force -ErrorAction SilentlyContinue
  }
}

# The seams a check replaces (`& (Get-Module Smoke) { $script:Hooks.Client = { ... } }`). The scripts never touch them.
$script:Hooks = @{
  Preflight   = { param($Model) Assert-Preflight -Model $Model }
  StartServer = {
    param($Exe, $Model, $Name, $RunDir, $KeyFile, $Config)
    $start = @{ Exe = $Exe; Model = $Model; Name = $Name; RunDir = $RunDir; KeyFile = $KeyFile }
    if ($Config) { $start.Config = $Config }
    Start-ArtemisServer @start
  }
  StopServer  = { param($Proc, $RunDir) Stop-ArtemisServer -Proc $Proc -RunDir $RunDir }
  Client      = { param($Root, $Client, $KeyFile, $ServerPid, $Project, $Out, $Task, $TimeoutS, $Repo, $Name) Invoke-SmokeCapture @PSBoundParameters }
  Tests       = { param($Root, $Project, $Repo, $Name) Invoke-SmokeTests @PSBoundParameters }
  BlankKey    = { param($Root, $Repo) Invoke-SmokeBlankKey @PSBoundParameters }
}

# --- public functions --------------------------------------------------------------------------------------------

# Refuses unless the smoke canary's marker for Client is at the fixed path and binds exactly the files as they are now.
# The same test as capture_clients.py's require_smoke_canary: the marker is JSON, its `client` is Client, and its `scripts`
# map holds the SHA-256 of each of the seven bound files by repo-relative path. A bound file that is missing throws.
function Assert-SmokeCanary {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][ValidateSet('opencode', 'claude-code')][string]$Client,
    [string]$Base = $script:SmokeBase,
    [string]$Repo = $script:RepoRoot
  )
  $path = Join-Path $Base "canary-ok-smoke-$Client"
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "canary-ok-smoke-$Client is missing: run smoke-canary first (capture_clients.py smoke-canary --client $Client); a smoke run does not start without it"
  }
  try {
    $data = [System.IO.File]::ReadAllText($path) | ConvertFrom-Json -AsHashtable
  } catch { throw "canary-ok-smoke-$Client is unreadable" }
  $current = Get-SmokeHashes $Repo
  $scripts = if ($data -is [hashtable]) { $data['scripts'] } else { $null }
  $same = ($data -is [hashtable]) -and ($data['client'] -ceq $Client) -and ($scripts -is [hashtable]) -and ($scripts.Count -eq $current.Count)
  if ($same) {
    foreach ($k in $current.Keys) {
      if (-not $scripts.ContainsKey($k) -or ([string]$scripts[$k] -cne $current[$k])) { $same = $false; break }
    }
  }
  if (-not $same) { throw "the bound files changed after the $Client smoke canary: run smoke-canary again" }
}

# The wire summary of a run: the server's request lines (Get-ChildItem LogDir\artemis-*.jsonl, lifecycle lines skipped).
# A generation line is a 200 on /v1/chat/completions or /v1/messages. `foreign` lists the lines that prove a credential
# other than the run's key reached the server: any 401, and any /v1/* line stamped strictly before the whole second of the
# client's launch (LaunchUtc; the log's time has second resolution). `unparsed` counts lines that are not a request line.
function Get-SmokeWire {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$LogDir,
    [Parameter(Mandatory)][datetime]$LaunchUtc
  )
  if (-not (Test-Path -LiteralPath $LogDir -PathType Container)) { throw "the server's log directory $LogDir does not exist" }
  $launch = Get-WholeSecond $LaunchUtc
  $repairs = [ordered]@{}
  $ignored = [ordered]@{}
  $wire = [ordered]@{
    requests = 0; streamed = 0; tool_calls = 0; malformed_calls = 0; status_5xx = 0; repairs = $repairs; ignored = $ignored
    status_other = 0; foreign = 0; unparsed = 0; lines = 0
  }
  $files = @(Get-ChildItem -LiteralPath $LogDir -Filter 'artemis-*.jsonl' -File | Sort-Object Name)
  foreach ($file in $files) {
    foreach ($raw in [System.IO.File]::ReadLines($file.FullName)) {
      if (-not $raw.Trim()) { continue }
      try { $j = $raw | ConvertFrom-Json } catch { $wire.unparsed++; continue }
      $names = @($j.PSObject.Properties.Name)
      if ($names -contains 'event') { continue }
      if (-not ($names -contains 'endpoint') -or -not ($names -contains 'status') -or -not ($names -contains 'time')) { $wire.unparsed++; continue }
      $wire.lines++
      $status = [int]$j.status
      $endpoint = [string]$j.endpoint
      $generation = ($endpoint -eq '/v1/chat/completions' -or $endpoint -eq '/v1/messages')
      $isForeign = ($status -eq 401)
      if (-not $isForeign -and $endpoint.StartsWith('/v1/', [System.StringComparison]::Ordinal)) {
        try { $isForeign = ((ConvertTo-LineTime $j.time) -lt $launch) } catch { $wire.unparsed++ }
      }
      if ($isForeign) { $wire.foreign++ }
      if ($status -ge 500) { $wire.status_5xx++ }
      if ($names -contains 'malformed_calls') { $wire.malformed_calls += [int]$j.malformed_calls }
      if ($names -contains 'repairs') { Add-Names $repairs $j.repairs }
      if ($names -contains 'ignored') { Add-Names $ignored $j.ignored }
      if ($generation -and $status -eq 200) {
        $wire.requests++
        if ($j.stream -eq $true) { $wire.streamed++ }
        if ($names -contains 'tool_calls') { $wire.tool_calls += [int]$j.tool_calls }
      } elseif ($generation -and $status -ne 401 -and $status -lt 500) {
        $wire.status_other++   # a 4xx or a 499 on a generation: listed, not gating
      }
    }
  }
  return $wire
}

# The verdict of a run, from the attempts, the wire summary and the harness's own failures (Failures: a dead server, an
# ungraceful stop). Each attempt is @{tests_pass; seconds; guard; hosts_status}. Returns @{status; reasons}.
#   fail          a guard's stop, a foreign credential, any 5xx, a malformed call, an unparsed log line, a Failures entry, a
#                 watch that could not vouch for where the client connected (hosts_status other than clean), or a wire log
#                 that is not clean (a generation not streamed, fewer than two tool calls) whatever the tests did
#   pass          the tests pass on some attempt, on a clean wire log
#   model-result  no attempt passes, on a clean wire log
function Get-SmokeStatus {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Attempts,
    [Parameter(Mandatory)]$Wire,
    [string[]]$Failures = @()
  )
  $reasons = [System.Collections.Generic.List[string]]::new()
  foreach ($f in $Failures) { $reasons.Add($f) }
  if (-not $Attempts.Count) { $reasons.Add('no attempt ran') }
  $n = 0
  foreach ($a in $Attempts) {
    $n++
    if ($a.guard) { $reasons.Add("attempt ${n}: the guard stopped the run: $($a.guard)") }
    elseif ($a.hosts_status -ne 'clean') { $reasons.Add("attempt ${n}: the watch could not vouch for where the client connected (hosts_status $($a.hosts_status))") }
  }
  if ($Wire.foreign -gt 0) { $reasons.Add("a credential other than the run's key reached the server ($($Wire.foreign) request line(s): a 401, or a /v1 request before the client's launch)") }
  if ($Wire.status_5xx -gt 0) { $reasons.Add("the server answered $($Wire.status_5xx) request(s) with a 5xx") }
  if ($Wire.malformed_calls -gt 0) { $reasons.Add("$($Wire.malformed_calls) malformed tool call(s)") }
  if ($Wire.unparsed -gt 0) { $reasons.Add("$($Wire.unparsed) log line(s) could not be read as request lines") }
  if ($reasons.Count) { return @{ status = 'fail'; reasons = @($reasons) } }
  # Not a server error: the run did not show the evidence the smoke exists for (streamed requests, at least two tool
  # calls).
  if ($Wire.requests -lt 1 -or $Wire.streamed -ne $Wire.requests) { $reasons.Add("the run did not show the required evidence (every generation streamed), not a server error: $($Wire.streamed) of $($Wire.requests) generation(s) streamed") }
  if ($Wire.tool_calls -lt 2) { $reasons.Add("the run did not show the required evidence (at least 2 tool calls), not a server error: $($Wire.tool_calls) tool call(s)") }
  if ($reasons.Count) { return @{ status = 'fail'; reasons = @($reasons) } }
  $passed = @($Attempts | Where-Object { $_.tests_pass }).Count -gt 0
  return @{ status = $(if ($passed) { 'pass' } else { 'model-result' }); reasons = @() }
}

# One smoke run of Client. -Config is a config template (tools/config/artemis-production.toml.in) the server starts from,
# as Start-ArtemisServer -Config does; without it the server runs on its flags alone. Writes <root>\verdict.json and
# returns it; throws (before anything is created) when the canary
# marker is missing or stale, the preflight fails or the root is wrong. The smoke base and the checkout are the module's own
# ($script:SmokeBase, $script:RepoRoot): the checks of this module set them from inside the module, a caller cannot, so the
# hashes checked are always those of the code that runs.
function Invoke-Smoke {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][ValidateSet('opencode', 'claude-code')][string]$Client,
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Model,
    [string]$Root,
    [ValidateRange(1, 2)][int]$MaxAttempts = 2,
    [string]$Config
  )
  $Base = Resolve-FullPath $script:SmokeBase
  $Repo = Resolve-FullPath $script:RepoRoot
  if (-not $Root) { $Root = Join-Path $Base "$Client-$(Get-Date -Format yyyyMMdd-HHmmss)" }
  $Root = Resolve-FullPath $Root
  if (-not (Test-SamePath (Split-Path $Root -Parent) $Base)) { throw "the run root $Root must be a direct child of $Base" }

  # Before anything is created or started: the canary, the preflight, the root and the keys.
  Assert-SmokeCanary -Client $Client -Base $Base -Repo $Repo
  foreach ($f in $script:ScratchFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Repo "tools\smoke\scratch\$f") -PathType Leaf)) { throw "the scratch repo is missing $f" }
  }
  if ($Config) { Assert-ConfigTemplate $Config }
  $pre = & $script:Hooks.Preflight $Model
  if ((Test-Path -LiteralPath $Root) -and @(Get-ChildItem -LiteralPath $Root -Force).Count) { throw "the run root $Root exists and is not empty" }
  $keysDir = Join-Path $Base 'keys'
  # A key file left by an earlier run of this client means its scan failed: that leak must be found first.
  $stale = @(Get-ChildItem -LiteralPath $keysDir -Directory -Filter "$Client-*" -ErrorAction SilentlyContinue |
      Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'key.txt') })
  if ($stale.Count) {
    throw "$($stale[0].FullName)\key.txt exists, so an earlier $Client smoke failed its key scan (or died before it). Check that run's root for a leak of the key, clean it, delete the key file and run again"
  }

  New-Item -ItemType Directory -Force $Root | Out-Null
  $serverDir = Join-Path $Root 'server'
  $logDir = Join-Path $serverDir 'logs'
  Copy-Scratch $Repo (Join-Path $Root 'project')
  $keyFile = New-ArtemisKey -Dir (Join-Path $keysDir (Split-Path $Root -Leaf))

  $attempts = [System.Collections.Generic.List[object]]::new()
  $failures = [System.Collections.Generic.List[string]]::new()
  $version = ''
  $launchFirst = $null
  $stop = $null
  $proc = $null
  $ranConfig = $null
  try {
    $proc = & $script:Hooks.StartServer $Exe $Model "smoke-$Client" $serverDir $keyFile $Config
    # What the server ran from (paths and hashes, never the key); a hook's stand-in process may carry none.
    $ranConfig = $proc.PSObject.Properties['ArtemisConfig']?.Value
    for ($n = 1; $n -le $MaxAttempts; $n++) {
      $projName = if ($n -eq 1) { 'project' } else { "project-$n" }
      if ($n -gt 1) { Copy-Scratch $Repo (Join-Path $Root $projName) }
      if ($proc.HasExited) { $failures.Add("the server died before attempt $n"); break }
      $launch = [datetime]::UtcNow
      if (-not $launchFirst) { $launchFirst = $launch }
      $out = Join-Path $Root "verdict-client-$n.json"
      $run = & $script:Hooks.Client $Root $Client $keyFile $proc.Id $projName $out $script:Task $script:ClientTimeoutS $Repo "capture-$n"
      $res = $run.result
      if ($res -and $res.version -and -not $version) { $version = [string]$res.version }
      $entry = [ordered]@{ tests_pass = $false; seconds = 0.0; guard = $null; hosts_status = $(if ($res) { [string]$res.hosts_status } else { 'inconclusive' }) }
      if ($res) { $entry.seconds = [double]$res.seconds; $entry.client_exit = $res.exit; $entry.timed_out = [bool]$res.timed_out }
      # Any stop, a launcher that did not exit 0, a missing result or no sign of the redirected home is a guard stop.
      if (-not $res) { $entry.guard = "capture_clients.py left no result (exit $($run.exit)$(if ($run.timed_out) { ', timed out' }))" }
      elseif ($res.stop) { $entry.guard = [string]$res.stop }
      elseif ($run.exit -ne 0) { $entry.guard = "capture_clients.py exited $($run.exit)" }
      elseif ($res.redirect_ok -ne $true) { $entry.guard = 'no sign the client used the redirected home' }
      elseif ($res.creds_unchanged -eq $false) { $entry.guard = "the credentials file changed" }
      if ($proc.HasExited) { $failures.Add("the server died during attempt $n") }
      if (-not $entry.guard -and -not $proc.HasExited) {
        $t = & $script:Hooks.Tests $Root $projName $Repo "tests-$n"
        $entry.tests_pass = [bool]$t.pass
        if ($t.note) { $entry.note = [string]$t.note }
      }
      $attempts.Add($entry)
      # A guard's stop, a watch that cannot vouch for the run or a dead server ends the run: the verdict is fail either way,
      # and a second attempt would only spend more of the engine's time. A pass ends it too.
      if ($entry.guard -or $entry.hosts_status -ne 'clean' -or $proc.HasExited -or $entry.tests_pass) { break }
    }
  } catch {
    $failures.Add("the run failed: $($_.Exception.Message)")
  } finally {
    # If a pipeline stop landed as the server started, $proc was never assigned: ask the module for the child.
    if (-not $proc) { $proc = Get-ArtemisServerProcess }
    if ($proc) {
      try { $stop = & $script:Hooks.StopServer $proc $serverDir } catch { $failures.Add("stopping the server failed: $($_.Exception.Message)") }
    }
  }
  if ($stop -and (-not $stop.graceful -or $stop.port_free -eq $false)) {
    $failures.Add("the server did not stop cleanly (graceful $($stop.graceful), exit code $($stop.exit_code), port free $($stop.port_free))")
  }

  $wire = $null
  try {
    $wire = Get-SmokeWire -LogDir $logDir -LaunchUtc $(if ($launchFirst) { $launchFirst } else { [datetime]::UtcNow })
  } catch {
    $failures.Add("the wire log could not be read: $($_.Exception.Message)")
    $wire = [ordered]@{ requests = 0; streamed = 0; tool_calls = 0; malformed_calls = 0; status_5xx = 0; repairs = [ordered]@{}; ignored = [ordered]@{}
      status_other = 0; foreign = 0; unparsed = 0; lines = 0 }
  }
  $judged = Get-SmokeStatus -Attempts $attempts.ToArray() -Wire $wire -Failures $failures.ToArray()

  $verdict = [ordered]@{
    client = $Client; version = $version; status = $judged.status; reasons = @($judged.reasons)
    attempts = @($attempts); wire = $wire
    server = $(if ($stop) { [ordered]@{ graceful = $stop.graceful; exit_code = $stop.exit_code; seconds = $stop.seconds } } else { $null })
    launch_utc = $(if ($launchFirst) { $launchFirst.ToString('o') } else { $null })
    preflight = $pre; root = $Root; config = $ranConfig
  }
  $verdictPath = Join-Path $Root 'verdict.json'
  Write-Utf8 $verdictPath (($verdict | ConvertTo-Json -Depth 8) + "`n")

  # The key scan runs over the whole root, whatever happened. OpenCode keeps the key in two temporary configs by design
  # (capture_clients.py's write_opencode_config): when a client run completed they must be there, holding it; when none
  # did (the launcher stopped before writing them) only those that exist are exempt. A Claude Code run has no exempt file.
  $expected = @()
  if ($Client -eq 'opencode') {
    $configs = @((Join-Path $Root 'home\opencode.json'), (Join-Path $Root 'home\.config\opencode\opencode.json'))
    $completed = @($attempts | Where-Object { -not $_.guard }).Count -gt 0
    # @(...) round the if: an empty result would otherwise be $null, which Assert-NoKey's string[] binds as one empty path.
    $expected = @(if ($completed) { $configs } else { $configs | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } })
  }
  $scanOk = $false
  try {
    $null = Assert-NoKey -KeyFile $keyFile -Paths $Root -Expected $expected
    $scanOk = $true
  } catch {
    $verdict.status = 'fail'
    $verdict.reasons = @($verdict.reasons) + "key scan: $($_.Exception.Message)"
    Write-Utf8 $verdictPath (($verdict | ConvertTo-Json -Depth 8) + "`n")
    Write-Warning "the key scan did not pass: the key file stays at $keyFile (its value is in no log); find the leak and clean it"
  } finally {
    if ($scanOk) {
      # After a clean scan the key has no more use: OpenCode's two configs are the only files that held it, so blank it
      # there too.
      if ($Client -eq 'opencode') {
        try { & $script:Hooks.BlankKey $Root $Repo } catch { Write-Warning "could not blank the key in OpenCode's configs under $Root`: $($_.Exception.Message)" }
      }
      try { Remove-Item -LiteralPath $keyFile -Force -ErrorAction Stop } catch { Write-Warning "could not delete the key file $keyFile`: $($_.Exception.Message)" }
    }
  }
  return [pscustomobject]$verdict
}

Export-ModuleMember -Function Assert-SmokeCanary, Get-SmokeWire, Get-SmokeStatus, Invoke-Smoke
