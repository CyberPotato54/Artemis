#Requires -Version 7.2
# The OpenCode smoke: the bundled opencode-cli.exe fixes the planted bugs in a scratch repo
# against `artemis serve` on 127.0.0.1:8090, which this script starts with a temporary key and stops. Refuses to start
# without the canary marker for these files (T:\artemis-oracle\runs\m3c-smoke\canary-ok-smoke-opencode). Writes
# <root>\verdict.json; exits 0 on pass or model-result and 1 on fail.
#   -Exe          the artemis.exe to serve
#   -Model        the GGUF
#   -Root         the run root, a new directory directly below T:\artemis-oracle\runs\m3c-smoke (default: opencode-<time>)
#   -MaxAttempts  1 or 2 (default 2: a failed first attempt is retried once in a fresh copy of the repo)
#   -Config       a config template (tools/config/artemis-production.toml.in) the server starts from (default: none, flags only)
param(
  [Parameter(Mandatory)][string]$Exe,
  [Parameter(Mandatory)][string]$Model,
  [string]$Root,
  [ValidateRange(1, 2)][int]$MaxAttempts = 2,
  [string]$Config
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Smoke.psm1') -Force
try {
  $run = @{ Client = 'opencode'; Exe = $Exe; Model = $Model; MaxAttempts = $MaxAttempts }
  if ($Root) { $run.Root = $Root }
  if ($Config) { $run.Config = $Config }
  $v = Invoke-Smoke @run
} catch {
  [Console]::Error.WriteLine("smoke (opencode): $($_.Exception.Message)")
  exit 1
}
Write-Host "smoke (opencode): $($v.status); $(@($v.attempts).Count) attempt(s); verdict at $(Join-Path $v.root 'verdict.json')"
foreach ($r in @($v.reasons)) { Write-Host "  $r" }
exit $(if ($v.status -in @('pass', 'model-result')) { 0 } else { 1 })
