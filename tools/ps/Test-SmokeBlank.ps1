#Requires -Version 7.2
# CPU check of Invoke-Smoke's last step (tools/smoke/Smoke.psm1): OpenCode's key is blanked after a clean key scan and
# only then. Invoke-Smoke runs with every seam replaced ($script:Hooks: no server, no client, no model), in a temporary
# smoke base, with the real key file, the real key scan and the real scratch repo copy. Cases:
#   1. opencode, clean scan: the BlankKey hook runs once, with the run root, while the key file still exists; the key
#      file is deleted afterwards.
#   2. opencode, the key leaked into a stray file: the scan fails, BlankKey does not run, the key file stays.
#   3. opencode, the client stopped before it wrote its configs while the attempt counted as run: the scan fails
#      (the expected configs are missing), BlankKey does not run.
#   4. claude-code, clean scan: BlankKey never runs, the key file is deleted.
# It starts no process, binds no port and touches nothing outside its temporary directory. Exit 0 when the checks pass,
# 1 when one fails.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$smoke = Import-Module (Join-Path $PSScriptRoot '..\smoke\Smoke.psm1') -Force -PassThru

$failures = @()
function Expect([bool]$Ok, [string]$What) {
  if ($Ok) { Write-Host "ok   $What" } else { Write-Host "FAIL $What"; $script:failures += $What }
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-smoke-blank-$([guid]::NewGuid().ToString('N'))"
$script:blankCalls = @()
$script:keyExistedAtBlank = $null
$script:leak = $false
$script:skipConfigs = $false

# The hooks. The client hook stands in for a completed client run: OpenCode's two temporary configs hold the key, as
# capture_clients.py's write_opencode_config leaves them.
& $smoke {
  $script:Hooks.Preflight = { param($Model) @{} }
  $script:Hooks.StartServer = {
    param($Exe, $Model, $Name, $RunDir, $KeyFile, $Config)
    New-Item -ItemType Directory -Force (Join-Path $RunDir 'logs') | Out-Null
    [pscustomobject]@{ Id = 1; HasExited = $false }
  }
  $script:Hooks.StopServer = { param($Proc, $RunDir) @{ graceful = $true; exit_code = 0; seconds = 0.0; port_free = $true } }
  $script:Hooks.Tests = { param($Root, $Project, $Repo, $Name) @{ pass = $true } }
}
$clientHook = {
  param($Root, $Client, $KeyFile, $ServerPid, $Project, $Out, $Task, $TimeoutS, $Repo, $Name)
  $key = [System.IO.File]::ReadAllText($KeyFile).Trim()
  if ($Client -eq 'opencode' -and -not $script:skipConfigs) {
    foreach ($rel in @('home\opencode.json', 'home\.config\opencode\opencode.json')) {
      $p = Join-Path $Root $rel
      New-Item -ItemType Directory -Force (Split-Path $p -Parent) | Out-Null
      [System.IO.File]::WriteAllText($p, "{`"apiKey`":`"$key`"}")
    }
  }
  if ($script:leak) { [System.IO.File]::WriteAllText((Join-Path $Root 'stray.txt'), "leaked $key") }
  @{ exit = 0; timed_out = $false; result = [pscustomobject]@{
      version = 'check'; seconds = 1.0; exit = 0; timed_out = $false; stop = $null
      hosts_status = 'clean'; redirect_ok = $true; creds_unchanged = $true } }
}
$blankHook = {
  param($Root, $Repo)
  $script:blankCalls += $Root
  $script:keyExistedAtBlank = @(Get-ChildItem -LiteralPath (Join-Path $script:base 'keys') -Recurse -Filter key.txt -File -ErrorAction SilentlyContinue).Count -eq 1
}
& $smoke { param($c, $b) $script:Hooks.Client = $c; $script:Hooks.BlankKey = $b } $clientHook $blankHook

# One run in its own smoke base (a stale key file of one case must not stop the next). Returns the root and key folder.
function Invoke-Case([string]$Client, [bool]$Leak, [bool]$SkipConfigs) {
  $script:base = Join-Path $tmp "$Client-$([guid]::NewGuid().ToString('N').Substring(0, 6))"
  New-Item -ItemType Directory -Force $script:base | Out-Null
  $script:blankCalls = @()
  $script:keyExistedAtBlank = $null
  $script:leak = $Leak
  $script:skipConfigs = $SkipConfigs
  & $smoke {
    param($base)
    $script:SmokeBase = $base
    $hashes = Get-SmokeHashes $script:RepoRoot
    foreach ($c in 'opencode', 'claude-code') {
      $marker = [ordered]@{ client = $c; scripts = $hashes }
      [System.IO.File]::WriteAllText((Join-Path $base "canary-ok-smoke-$c"), ($marker | ConvertTo-Json -Depth 4))
    }
  } $script:base
  $root = Join-Path $script:base "run-$Client"
  $null = Invoke-Smoke -Client $Client -Exe 'unused' -Model 'unused' -Root $root -MaxAttempts 1 -WarningAction SilentlyContinue
  $keys = @(Get-ChildItem -LiteralPath (Join-Path $script:base 'keys') -Recurse -Filter key.txt -File -ErrorAction SilentlyContinue)
  return @{ root = $root; keyFiles = $keys.Count }
}

try {
  $r = Invoke-Case 'opencode' $false $false
  Expect ($script:blankCalls.Count -eq 1 -and $script:blankCalls[0] -eq $r.root) 'opencode, clean scan: the key is blanked once, for the run root'
  Expect ($script:keyExistedAtBlank -eq $true) 'opencode, clean scan: the blank runs while the key file still exists'
  Expect ($r.keyFiles -eq 0) 'opencode, clean scan: the key file is deleted afterwards'

  $r = Invoke-Case 'opencode' $true $false
  Expect ($script:blankCalls.Count -eq 0) 'opencode, key found in a stray file: the key is not blanked'
  Expect ($r.keyFiles -eq 1) 'opencode, key found in a stray file: the key file stays for the leak hunt'

  $r = Invoke-Case 'opencode' $false $true
  Expect ($script:blankCalls.Count -eq 0) 'opencode, expected configs missing (scan fails): the key is not blanked'
  Expect ($r.keyFiles -eq 1) 'opencode, expected configs missing (scan fails): the key file stays'

  $r = Invoke-Case 'claude-code' $false $false
  Expect ($script:blankCalls.Count -eq 0) 'claude-code, clean scan: the blank is never called'
  Expect ($r.keyFiles -eq 0) 'claude-code, clean scan: the key file is deleted'
} catch {
  Write-Host "FAIL the check threw: $($_.Exception.Message)"
  $failures += 'the check threw'
} finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count) { Write-Host "$($failures.Count) check(s) failed"; exit 1 }
Write-Host 'all checks passed'
exit 0
