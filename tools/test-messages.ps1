#Requires -Version 7.2
# Test-Messages: a three-step Anthropic tool-use check against a running
# server. Our own script, written for Artemis.
#   1  an unauthenticated `{}` gets 401 in Anthropic's error shape
#   2  a request with a read_file tool answers with a tool_use block (name read_file, input.path), stop_reason tool_use
#   3  step 2's content echoed verbatim, with a tool_result, answers with a text block and end_turn; the prompt's
#      reused part shows as usage.cache_read_input_tokens, which must be above 0 and is printed
# The key is read from -KeyFile into a variable and sent as `x-api-key`; it is never printed or logged, and it is never on
# a command line. -BaseUrl must be http://127.0.0.1:PORT or http://localhost:PORT (no https, no path), and not port 8080
# (the production server listens there): anything else is refused before any request, as the test-7 driver refuses it. One line per step; exit 0 when all three pass, 1 on a failed assertion or a refusal.
param(
  [string]$BaseUrl = 'http://127.0.0.1:8090',
  [Parameter(Mandatory)][string]$KeyFile
)
$ErrorActionPreference = 'Stop'

function Fail([string]$Line) {
  Write-Host $Line
  exit 1
}

# --- the refusal, before anything else -----------------------------------------------------------------------------
# The driver's rules (`check_base_url` in crates/tools/src/test7/mod.rs): `http://` only, then host:port with one trailing
# `/` allowed and no path, user, query or fragment. A refusal never echoes the URL beyond the host.
if (-not $BaseUrl.StartsWith('http://', [System.StringComparison]::Ordinal)) {
  Fail "refused: -BaseUrl must start with http://"
}
$rest = $BaseUrl.Substring('http://'.Length)
if ($rest.EndsWith('/')) { $rest = $rest.Substring(0, $rest.Length - 1) }
if ($rest -eq '' -or $rest.IndexOfAny([char[]]@('/', '@', '?', '#', '\', ' ')) -ge 0) {
  Fail "refused: -BaseUrl must be http://127.0.0.1:PORT or http://localhost:PORT, with no path or user"
}
if ($rest -notmatch '^[^:]+:[0-9]+$') {
  Fail "refused: -BaseUrl names no port, or one that is not a number"
}
$uri = $null
if (-not [uri]::TryCreate("http://$rest", [System.UriKind]::Absolute, [ref]$uri)) {
  Fail "refused: -BaseUrl is not a valid URL (its port may be out of range)"
}
if ($uri.Host -notin @('127.0.0.1', 'localhost')) {
  Fail "refused: -BaseUrl host '$($uri.Host)' is not 127.0.0.1 or localhost"
}
if ($uri.Port -eq 8080) {
  Fail "refused: -BaseUrl port 8080 is the production server's"
}
$base = "http://$rest"

if (-not (Test-Path -LiteralPath $KeyFile -PathType Leaf)) {
  Fail "-KeyFile does not name a file"
}
# The first non-empty line, trimmed, as the server and ArtemisServer.psm1's Read-Key take it. Interpolated, so an empty
# file (Get-Content -Raw gives no output at all) is '' and not $null.
$key = ("$(Get-Content -LiteralPath $KeyFile -Raw)".TrimStart([char]0xFEFF).Split("`n") | ForEach-Object { $_.Trim() } |
    Where-Object { $_ } | Select-Object -First 1)
if (-not $key) {
  Fail "-KeyFile is empty"
}

$model = 'claude-sonnet-4-5'

# POSTs $Body (JSON text) to /v1/messages; the status and the body text. No key header when -NoKey.
function Send-Messages([string]$Body, [switch]$NoKey) {
  $headers = @{ 'anthropic-version' = '2023-06-01' }
  if (-not $NoKey) { $headers['x-api-key'] = $key }
  $r = Invoke-WebRequest -Uri "$base/v1/messages" -Method Post -Headers $headers -ContentType 'application/json' `
    -Body $Body -SkipHttpErrorCheck -TimeoutSec 600
  return [pscustomobject]@{ Status = [int]$r.StatusCode; Text = [string]$r.Content }
}

# The error type of a non-200 reply, for a failure line (never the message: it can echo request text).
function Error-Type($Reply) {
  try { return [string]((ConvertFrom-Json $Reply.Text).error.type) } catch { return '' }
}

$step = 0
try {
  # --- step 1: no credentials ------------------------------------------------------------------------------------
  $step = 1
  $r = Send-Messages '{}' -NoKey
  if ($r.Status -ne 401) { Fail "step 1: FAIL expected 401 on an unauthenticated {}, got $($r.Status)" }
  $e = ConvertFrom-Json $r.Text
  if ($e.type -ne 'error' -or $e.error.type -ne 'authentication_error') {
    Fail "step 1: FAIL the 401 is not Anthropic's authentication_error shape"
  }
  Write-Host "step 1: PASS 401 authentication_error"

  # --- step 2: a tool_use block --------------------------------------------------------------------------------
  $step = 2
  $system = 'You can read files with read_file.'
  $tools = ConvertTo-Json -InputObject @(, [ordered]@{
      name         = 'read_file'
      description  = 'Reads a file.'
      input_schema = [ordered]@{
        type       = 'object'
        properties = [ordered]@{ path = [ordered]@{ type = 'string' } }
        required   = @('path')
      }
    }) -Depth 10 -Compress
  $question = ConvertTo-Json -InputObject ([ordered]@{
      role    = 'user'
      content = "What is the project's name? Read pyproject.toml to find out."
    }) -Depth 10 -Compress
  $head = '{"model":' + (ConvertTo-Json $model) + ',"max_tokens":1024,"temperature":0,"system":' +
  (ConvertTo-Json $system) + ',"tools":' + $tools + ',"messages":'

  $r = Send-Messages ($head + '[' + $question + ']}')
  if ($r.Status -ne 200) { Fail "step 2: FAIL expected 200, got $($r.Status) $(Error-Type $r)" }
  $b2 = ConvertFrom-Json $r.Text
  $call = @($b2.content | Where-Object { $_.type -eq 'tool_use' })
  if ($call.Count -ne 1) { Fail "step 2: FAIL expected one tool_use block, got $($call.Count) (stop_reason $($b2.stop_reason))" }
  if ($call[0].name -ne 'read_file') { Fail "step 2: FAIL the tool_use is named '$($call[0].name)', not read_file" }
  if ($call[0].input.path -isnot [string] -or -not $call[0].input.path) { Fail "step 2: FAIL the tool_use has no input.path" }
  if (-not $call[0].id) { Fail "step 2: FAIL the tool_use has no id" }
  if ($b2.stop_reason -ne 'tool_use') { Fail "step 2: FAIL stop_reason is '$($b2.stop_reason)', not tool_use" }
  Write-Host "step 2: PASS tool_use read_file, stop_reason tool_use"

  # --- step 3: the echo and the tool_result --------------------------------------------------------------------
  $step = 3
  # Step 2's content goes back verbatim (thinking block included): re-read as JSON nodes and written back unchanged,
  # not through PowerShell objects, which would change the text of anything date-shaped.
  $content = [System.Text.Json.Nodes.JsonNode]::Parse($r.Text)['content'].ToJsonString()
  $assistant = '{"role":"assistant","content":' + $content + '}'
  $result = ConvertTo-Json -InputObject ([ordered]@{
      role    = 'user'
      content = @(, [ordered]@{
          type        = 'tool_result'
          tool_use_id = [string]$call[0].id
          content     = "[project]`nname = `"example`"`nversion = `"0.1.0`"`n"
        })
    }) -Depth 10 -Compress

  $r = Send-Messages ($head + '[' + $question + ',' + $assistant + ',' + $result + ']}')
  if ($r.Status -ne 200) { Fail "step 3: FAIL expected 200, got $($r.Status) $(Error-Type $r)" }
  $b3 = ConvertFrom-Json $r.Text
  $text = @($b3.content | Where-Object { $_.type -eq 'text' -and $_.text })
  if ($text.Count -lt 1) { Fail "step 3: FAIL no text block in the answer" }
  if ($b3.stop_reason -ne 'end_turn') { Fail "step 3: FAIL stop_reason is '$($b3.stop_reason)', not end_turn" }
  $cached = $b3.usage.cache_read_input_tokens
  if ($cached -isnot [int] -and $cached -isnot [long]) { Fail "step 3: FAIL usage has no cache_read_input_tokens" }
  if ($cached -le 0) { Fail "step 3: FAIL cache_read_input_tokens is $cached, expected above 0" }
  Write-Host "step 3: PASS text block, stop_reason end_turn"
  Write-Host "cache_read_input_tokens: $cached"
} catch {
  # A transport error or a reply that is not JSON: the step and the exception type only (a message can quote the reply).
  Fail "step ${step}: FAIL $($_.Exception.GetType().Name)"
}
