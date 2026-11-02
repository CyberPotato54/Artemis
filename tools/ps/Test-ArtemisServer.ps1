#Requires -Version 7.2
# CPU check of Start-ArtemisServer -Config, up to and past the config fill, without starting artemis (a regression
# check: PowerShell variable names are case-insensitive, and a local `$config` once overwrote the -Config parameter, so
# the fill read the run's own artemis.toml as its template). Two cases, each in its own temporary directory, with a
# dummy model and a generated key file:
#   1. The stub executable is a .cmd that exits at once (code 3). The function fills the template and starts the stub,
#      and throws "exited with code ... before /health answered". The run's artemis.toml must hold the filled template:
#      port 8090, the dummy model, key file and log directory, no placeholder left.
#   2. The stub is a .cmd that waits on its stdin (the stop signal), and the module's Get-Health and Get-PortOwner are
#      replaced inside the module so the stub looks like a healthy server. The returned Process must carry
#      ArtemisConfig.template = the template's path (not the run's artemis.toml) with the hashes of both files.
# Two checks before the two cases cover New-ArtemisKey's owner-only access rules and Invoke-Smoke's key blanking
# (below). It touches nothing outside its temporary directory, binds no port, and starts no artemis. Port 8090 must be
# free and no artemis process may run (the function's own refusals); if either is not so, the two cases are skipped, the
# key checks still count, and the exit is 0 unless one failed. Exit 0 when the checks pass, 1 when one fails.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$template = Join-Path $repo 'tools\config\artemis-production.toml.in'
$module = Import-Module (Join-Path $PSScriptRoot 'ArtemisServer.psm1') -Force -PassThru

$failures = @()
function Expect([bool]$Ok, [string]$What) {
  if ($Ok) { Write-Host "ok   $What" } else { Write-Host "FAIL $What"; $script:failures += $What }
}

# 0. The key file is owner-only and the owner can still use it. This runs before the skips below: it starts no process
# and binds no port. The file's access rules are protected (no inheritance) and exactly one rule remains, for the
# current user; that user can read the key (Read-Key, which Assert-NoKey uses), write the file, and delete it (the
# smoke removes the key file after a clean scan). The folder is owner-only the same way.
$keyBase = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-key-check-$([guid]::NewGuid().ToString('N'))"
try {
  $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $keyPath = New-ArtemisKey -Dir (Join-Path $keyBase 'keys\one')
  foreach ($item in @(@('the key file', $keyPath), @('the key folder', (Split-Path $keyPath -Parent)))) {
    $acl = Get-Acl -LiteralPath $item[1]
    $rules = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
    Expect $acl.AreAccessRulesProtected "$($item[0]) does not inherit access rules"
    Expect ($rules.Count -eq 1) "$($item[0]) has exactly one access rule (found $($rules.Count))"
    Expect ($rules.Count -eq 1 -and $rules[0].IdentityReference.Value -eq $me.Value -and $rules[0].AccessControlType -eq 'Allow') "$($item[0])'s rule allows the current user"
  }
  $read = $null
  try { $read = & $module { param($f) Read-Key $f } $keyPath } catch { $read = $null }
  Expect ($read -and $read.Length -eq 43) 'the current user reads the key back (Read-Key)'
  $wrote = $false
  try {
    [System.IO.File]::AppendAllText($keyPath, "`n")
    [System.IO.File]::WriteAllText($keyPath, "$read`n", [System.Text.UTF8Encoding]::new($false))
    $wrote = ([System.IO.File]::ReadAllText($keyPath) -eq "$read`n")
  } catch { }
  Expect $wrote 'the current user writes the key file'
  $again = New-ArtemisKey -Dir (Split-Path $keyPath -Parent)
  Expect ($again -eq $keyPath -and ([System.IO.File]::ReadAllText($again).Trim().Length -eq 43)) 'a second New-ArtemisKey in the same folder replaces the key'
  $deleted = $false
  try { Remove-Item -LiteralPath $keyPath -Force -ErrorAction Stop; $deleted = -not (Test-Path -LiteralPath $keyPath) } catch { }
  Expect $deleted 'the current user deletes the key file'
} finally {
  Remove-Item -LiteralPath $keyBase -Recurse -Force -ErrorAction SilentlyContinue
}

# 0b. Invoke-Smoke blanks OpenCode's key after a clean key scan and only then (Test-SmokeBlank.ps1, in its own process
# because it re-imports the modules).
$blankOut = & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'Test-SmokeBlank.ps1') 2>&1 | Out-String
$blankOut.TrimEnd() -split "`r?`n" | ForEach-Object { Write-Host "     $_" }
Expect ($LASTEXITCODE -eq 0) 'Invoke-Smoke blanks the OpenCode key after a clean scan only (Test-SmokeBlank.ps1)'

function Skip([string]$Why) {
  Write-Host "SKIP $Why"
  if ($failures.Count) { Write-Host "$($failures.Count) check(s) failed"; exit 1 }
  exit 0
}
if (@(Get-Process -Name artemis -ErrorAction SilentlyContinue).Count) { Skip 'an artemis process is running' }
if (& $module { Test-PortBusy $script:Port }) { Skip 'port 8090 is busy' }

$base = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-server-check-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $base | Out-Null
try {
  $modelFile = Join-Path $base 'model.gguf'
  [System.IO.File]::WriteAllBytes($modelFile, [byte[]](1, 2, 3))
  $keyFile = New-ArtemisKey -Dir (Join-Path $base 'key')
  $templateSha = [System.Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData([System.IO.File]::ReadAllBytes($template))).ToLowerInvariant()

  # 1. The stub exits at once: the fill happened, the start did not get far.
  $exitStub = Join-Path $base 'exit.cmd'
  [System.IO.File]::WriteAllText($exitStub, "@echo off`r`nexit /b 3`r`n")
  $runDir = Join-Path $base 'run-exit'
  $message = $null
  try {
    $null = Start-ArtemisServer -Exe $exitStub -Model $modelFile -Name 'check-exit' -RunDir $runDir -KeyFile $keyFile -Config $template
  } catch { $message = $_.Exception.Message }
  Expect ($message -like '*artemis serve exited with code 3 before /health answered*') "the stub's early exit is what stopped the start ($message)"
  Expect ($message -notlike '*does not exist*') 'the template was found'
  $filledFile = Join-Path $runDir 'artemis.toml'
  $filled = Get-Content -LiteralPath $filledFile -Raw
  Expect ($filled -match '(?m)^port\s*=\s*8090\s*$') 'the run''s artemis.toml names port 8090'
  Expect ($filled.Contains((Join-Path $runDir 'logs'))) 'it names the run''s log directory'
  Expect ($filled.Contains($modelFile) -and $filled.Contains($keyFile)) 'it names the dummy model and key file'
  Expect (-not $filled.Contains('{{')) 'no placeholder is left'
  Expect ($filled -ne [System.IO.File]::ReadAllText($template)) 'it is not the template itself'
  Expect (@(Get-Process -Name artemis -ErrorAction SilentlyContinue).Count -eq 0) 'no artemis process was started'

  # 2. The stub waits on its stdin and the module is told it is healthy: the summary the Process carries.
  $stub = Join-Path $base 'stub.cmd'
  [System.IO.File]::WriteAllText($stub, "@echo off`r`nset /p x=`r`n")
  & $module {
    function script:Get-Health { [pscustomobject]@{ Status = 200; Body = '{}' } }
    function script:Get-PortOwner([int]$Port) { @($script:Live.Id) }
  }
  $runDir = Join-Path $base 'run-ok'
  $proc = $null
  try {
    $proc = Start-ArtemisServer -Exe $stub -Model $modelFile -Name 'check-ok' -RunDir $runDir -KeyFile $keyFile -Config $template
    $summary = $proc.ArtemisConfig
    $runToml = Join-Path $runDir 'artemis.toml'
    $runSha = [System.Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData([System.IO.File]::ReadAllBytes($runToml))).ToLowerInvariant()
    Expect ($summary.template -eq $template) 'ArtemisConfig.template is the template path'
    Expect ($summary.template -ne $runToml) 'it is not the run''s own artemis.toml'
    Expect ($summary.template_sha256 -eq $templateSha) 'ArtemisConfig.template_sha256 is the template''s hash'
    Expect ($summary.sha256 -eq $runSha) 'ArtemisConfig.sha256 is the run''s artemis.toml hash'
    Expect ($summary.overrides.port -eq 8090 -and $summary.overrides.model -eq $modelFile -and $summary.overrides.api_key_file -eq $keyFile) 'overrides hold the port, model and key file path'
    Expect ((Get-Content -LiteralPath $runToml -Raw).Contains($modelFile)) 'the run''s artemis.toml is the filled template'
  } finally {
    if ($proc) {
      try { $proc.StandardInput.Close() } catch { }
      if (-not $proc.WaitForExit(5000)) { try { $proc.Kill($true) } catch { } }
      & $module { param($p) Close-Drain $p } $proc
      $proc.Dispose()
    }
  }
} finally {
  Remove-Module ArtemisServer -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
}
if ($failures.Count) { Write-Host "$($failures.Count) check(s) failed"; exit 1 }
Write-Host 'all checks passed'
exit 0
