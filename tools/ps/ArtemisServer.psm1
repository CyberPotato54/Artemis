#Requires -Version 7.2
# Starts and stops `artemis serve` for the runner's server blocks and the client smoke scripts. The rules it enforces:
#   - 127.0.0.1:8090 only, never 8080 (another local service may hold it), and only when nothing else holds 8090 or runs
#     as `artemis`;
#   - always an explicit --config (the user's %APPDATA%\artemis\artemis.toml never applies) and its own --log-dir;
#   - the key goes in a file, never on a command line, and Assert-NoKey proves it reached no file it shouldn't;
#   - the child's stdin is a pipe this module holds open: `artemis serve --stop-on-stdin-close` stops when it closes, so
#     Stop-ArtemisServer closes it, and so does the OS if the script that started the server dies.
$ErrorActionPreference = 'Stop'

$script:Port = 8090
$script:Http = [System.Net.Http.HttpClient]::new()
$script:Http.Timeout = [TimeSpan]::FromSeconds(5)
# pid -> @{ Task; Stream }: the stderr copy of each child Start-Child started (see Start-Child).
$script:Drains = @{}
# The newest child Start-Child started and nobody has stopped yet. It is set inside Start-Child, before the caller has
# the Process in a variable, so a pipeline stop between the two can still find the child (Start-ArtemisServer's
# finally and Get-ArtemisServerProcess use it).
$script:Live = $null

# --- private helpers ---------------------------------------------------------------------------------------------

function Resolve-FullPath([string]$Path) {
  return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

# The pids of the processes listening on a TCP port (any address), from the listener table. Empty when none.
function Get-PortOwner([int]$Port) {
  return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique)
}

# Whether something listens on the port or accepts a connection to 127.0.0.1 on it.
function Test-PortBusy([int]$Port) {
  $listening = @([System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
      Where-Object { $_.Port -eq $Port })
  if ($listening.Count) { return $true }
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $connect = $client.ConnectAsync('127.0.0.1', $Port)
    return ($connect.Wait(500) -and $client.Connected)
  } catch { return $false } finally { $client.Dispose() }
}

# /health on 127.0.0.1:8090 as @{Status; Body}, or $null when nothing answers.
function Get-Health {
  try {
    $r = $script:Http.GetAsync("http://127.0.0.1:$($script:Port)/health").GetAwaiter().GetResult()
    try {
      return [pscustomobject]@{ Status = [int]$r.StatusCode; Body = $r.Content.ReadAsStringAsync().GetAwaiter().GetResult() }
    } finally { $r.Dispose() }
  } catch { return $null }
}

# Starts Exe with ArgList as a child whose stdin is a redirected pipe, held open (closing it is the stop; a null or
# closed stdin reads EOF at once and stops the server) and whose stderr is copied to StderrFile; stdout is inherited.
# The copy is a Stream.CopyToAsync on the thread pool, not Register-ObjectEvent: an event action runs on the script's
# own pipeline thread, so it starves while the runner blocks in a native command (`& artemis test7`, minutes long),
# the pipe fills, and the server's tracing writes block a worker. Returns the Process.
function Start-Child {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][Alias('Args')][string[]]$ArgList,
    [Parameter(Mandatory)][string]$StderrFile
  )
  $psi = [System.Diagnostics.ProcessStartInfo]::new($Exe)
  foreach ($a in $ArgList) { $psi.ArgumentList.Add($a) }
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardError = $true
  $proc = [System.Diagnostics.Process]::new()
  $proc.StartInfo = $psi
  # Unbuffered (bufferSize 1) and shared for reading, so the tail is readable while the child runs.
  $stream = [System.IO.FileStream]::new($StderrFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1)
  $done = $false
  try {
    $null = $proc.Start()
    $task = $proc.StandardError.BaseStream.CopyToAsync($stream)
    $script:Drains[$proc.Id] = @{ Task = $task; Stream = $stream }
    $script:Live = $proc
    $done = $true
  } finally {
    # Not done: a failed start, or a pipeline stop at any point after Start() began. Never leave the child running.
    # Kill on a process that never started throws, and the catch absorbs it; so there is no flag to set between
    # Start() and the kill.
    if (-not $done) {
      try { $proc.Kill($true) } catch { }
      try { $script:Drains.Remove($proc.Id) } catch { }
      $script:Live = $null
      $stream.Dispose()
    }
  }
  return $proc
}

# Waits for a child's stderr copy to finish (its pipe closes when it exits) and closes the file.
function Close-Drain([System.Diagnostics.Process]$Proc) {
  if ($script:Live -and $script:Live.Id -eq $Proc.Id) { $script:Live = $null }
  $d = $script:Drains[$Proc.Id]
  if (-not $d) { return }
  $script:Drains.Remove($Proc.Id)
  try { $null = $d.Task.Wait(5000) } catch { }
  $d.Stream.Dispose()
}

# Kills a child this module started (the whole tree) and releases it. Safe on an exited process.
function Stop-Child([System.Diagnostics.Process]$Proc) {
  try { if (-not $Proc.HasExited) { $Proc.Kill($true); $null = $Proc.WaitForExit(10000) } } catch { }
  Close-Drain $Proc
  $Proc.Dispose()
}

if (-not ('ArtemisNative.LongPath' -as [type])) {
  Add-Type -Namespace ArtemisNative -Name LongPath -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public static extern uint GetLongPathName(string shortPath, System.Text.StringBuilder longPath, uint size);'
}

# A path in one spelling: absolute, with 8.3 short names expanded when the file exists, so two names for one file compare
# equal (Assert-NoKey's -Expected against what the directory walk finds).
function Get-CanonicalPath([string]$Path) {
  $full = [System.IO.Path]::GetFullPath((Resolve-FullPath $Path))
  if (-not (Test-Path -LiteralPath $full)) { return $full }
  $sb = [System.Text.StringBuilder]::new(32768)
  $n = [ArtemisNative.LongPath]::GetLongPathName($full, $sb, [uint32]$sb.Capacity)
  if ($n -gt 0 -and $n -lt $sb.Capacity) { return $sb.ToString() }
  return $full
}

function Get-Tail([string]$File, [int]$Lines = 20) {
  if (-not (Test-Path -LiteralPath $File)) { return '(no stderr file)' }
  return ((Get-Content -LiteralPath $File -Tail $Lines) -join "`n")
}

# Whether the file holds the needle as ASCII/UTF-8 or as UTF-16LE bytes. Chunked, so a big file is fine; shared for
# reading, so a log the server still has open is too.
function Test-FileHolds([string]$Path, [string]$Needle) {
  $needles = @($Needle, ($Needle.ToCharArray() -join "`0"))
  $max = ($needles | Measure-Object -Property Length -Maximum).Maximum
  $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
  try {
    $buf = [byte[]]::new(1MB + $max)
    $keep = 0
    while ($true) {
      $n = $fs.Read($buf, $keep, $buf.Length - $keep)
      if ($n -le 0) { return $false }
      $len = $keep + $n
      # Latin-1 maps one byte to one char, so a byte search is a string search.
      $text = [System.Text.Encoding]::Latin1.GetString($buf, 0, $len)
      foreach ($x in $needles) {
        if ($text.Contains($x, [System.StringComparison]::Ordinal)) { return $true }
      }
      $keep = [Math]::Min($max - 1, $len)
      [Array]::Copy($buf, $len - $keep, $buf, 0, $keep)
    }
  } finally { $fs.Dispose() }
}

# The first non-empty line of the key file, trimmed. Never printed.
function Read-Key([string]$KeyFile) {
  if (-not (Test-Path -LiteralPath $KeyFile -PathType Leaf)) { throw "key file $KeyFile does not exist" }
  $line = [System.IO.File]::ReadAllText($KeyFile).Split("`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -First 1
  if (-not $line -or $line.Length -lt 32) { throw "key file $KeyFile does not hold a key" }
  return $line
}

# --- public functions --------------------------------------------------------------------------------------------

# Writes 32 random bytes as base64url (43 characters, no padding) plus "\n" (UTF-8, no BOM) to Dir\key.txt and returns
# the path. The value is never returned or printed.
# The file, and the folder when this call creates it, are owner-only: access rules not inherited, and one rule left,
# which allows the current user Modify (read, write and delete, so the owner can still read the key, replace it and
# remove it after a scan; read and write alone would block the delete). Each object is created with its security
# descriptor in one step (FileSystemAclExtensions.Create, .NET 6 and later, which is what PowerShell 7.2 runs on), so
# neither exists for a moment under an inherited ACL and no other user can open a handle in between. Any older key.txt
# is removed first and the file is created with CreateNew.
function New-ArtemisKey {
  [CmdletBinding()]
  param([Parameter(Mandatory)][string]$Dir)
  $Dir = Resolve-FullPath $Dir
  $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $inherit = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
  $none = [System.Security.AccessControl.PropagationFlags]::None
  $allow = [System.Security.AccessControl.AccessControlType]::Allow
  if (-not (Test-Path -LiteralPath $Dir -PathType Container)) {
    $dirAcl = [System.Security.AccessControl.DirectorySecurity]::new()
    $dirAcl.SetAccessRuleProtection($true, $false)
    $dirAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($me, 'FullControl', $inherit, $none, $allow))
    [System.IO.FileSystemAclExtensions]::Create([System.IO.DirectoryInfo]::new($Dir), $dirAcl) | Out-Null
  }
  $bytes = [System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
  $key = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
  $path = Join-Path $Dir 'key.txt'
  Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  $fileAcl = [System.Security.AccessControl.FileSecurity]::new()
  $fileAcl.SetAccessRuleProtection($true, $false)
  $fileAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($me, 'Modify', $allow))
  $stream = [System.IO.FileSystemAclExtensions]::Create([System.IO.FileInfo]::new($path), [System.IO.FileMode]::CreateNew,
    [System.Security.AccessControl.FileSystemRights]::Modify, [System.IO.FileShare]::None, 4096, [System.IO.FileOptions]::None, $fileAcl)
  try {
    $line = [System.Text.UTF8Encoding]::new($false).GetBytes("$key`n")
    $stream.Write($line, 0, $line.Length)
  } finally { $stream.Dispose() }
  return $path
}

# Checks that this machine can run the real model with a 32 GB machine's pool (19.2 GiB pinned + a 2 GiB pool + a
# 4 GiB margin = about 26 GiB free), that the model file is there and that the 9070 is. Returns the figures.
function Assert-Preflight {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Model,
    [double]$MinFreeGiB = 26
  )
  $freeGiB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
  if ($freeGiB -lt $MinFreeGiB) { throw "preflight: $freeGiB GiB of RAM free, $MinFreeGiB needed" }
  if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "preflight: the model file $Model does not exist" }
  $gpu = Get-CimInstance Win32_VideoController | Where-Object Name -Match '9070' | Select-Object -First 1
  if (-not $gpu) { throw 'preflight: no 9070 display adapter' }
  return @{
    free_ram_gib = $freeGiB; min_free_gib = $MinFreeGiB
    model = $Model; model_gib = [math]::Round((Get-Item -LiteralPath $Model).Length / 1GB, 2)
    adapter = $gpu.Name; driver = $gpu.DriverVersion
  }
}

# Fills a config template (tools/config/artemis-production.toml.in) and writes the result to Out (UTF-8, no BOM). The
# rules are the ones `fill_template` in crates/server/src/config.rs applies: each {{NAME}} becomes its value (MODEL,
# PORT, API_KEY_FILE, LOG_DIR, in that order); a value that holds a single quote, a line break (LF or CR) or {{ is
# refused, as is a placeholder the template lacks and any {{ still open after filling. Its messages are fill_template's,
# word for word, and name the placeholder, never a value. Beyond them: -Port must be 8090 (the runner never binds
# another port, and another local service may hold 8080), and the filled text may not name 8080 anywhere. Nothing is
# written on a refusal. Returns @{ path; sha256; template_sha256 } (lowercase hex of the file as written and of the
# template). The key file is a path here, never read.
function New-ArtemisConfig {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Template,
    [Parameter(Mandatory)][string]$Out,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$KeyFile,
    [Parameter(Mandatory)][string]$LogDir,
    [int]$Port = 8090
  )
  if ($Port -ne $script:Port) { throw "New-ArtemisConfig: the port must be $($script:Port); no other port is filled" }
  $Template = Resolve-FullPath $Template
  $Out = Resolve-FullPath $Out
  if (-not (Test-Path -LiteralPath $Template -PathType Leaf)) { throw "New-ArtemisConfig: the config template $Template does not exist" }
  $templateBytes = [System.IO.File]::ReadAllBytes($Template)
  $utf8 = [System.Text.UTF8Encoding]::new($false)
  $text = $utf8.GetString($templateBytes)
  $values = [ordered]@{ MODEL = $Model; PORT = "$Port"; API_KEY_FILE = $KeyFile; LOG_DIR = $LogDir }
  foreach ($name in @($values.Keys)) {
    # A path value is made absolute; the .NET error for a bad path would quote it, so a failure here says only the name.
    if ($name -ne 'PORT') {
      try { $values[$name] = Resolve-FullPath $values[$name] } catch { throw "the value for $name is not a usable path" }
    }
  }
  foreach ($name in $values.Keys) {
    $value = $values[$name]
    if ($value.Contains("'", [System.StringComparison]::Ordinal)) { throw "the value for $name holds a single quote, which a TOML literal string cannot hold" }
    if ($value.Contains("`n", [System.StringComparison]::Ordinal) -or $value.Contains("`r", [System.StringComparison]::Ordinal)) { throw "the value for $name holds a line break" }
    if ($value.Contains('{{', [System.StringComparison]::Ordinal)) { throw "the value for $name holds an opening double brace" }
    $placeholder = '{{' + $name + '}}'
    if (-not $text.Contains($placeholder, [System.StringComparison]::Ordinal)) { throw "the template has no placeholder $name" }
    $text = $text.Replace($placeholder, $value)
  }
  $at = $text.IndexOf('{{', [System.StringComparison]::Ordinal)
  if ($at -ge 0) {
    $rest = $text.Substring($at + 2)
    $open = ($rest -split "\r?\n", 2)[0]
    throw "no value for placeholder $(($open -split '}}', 2)[0])"
  }
  if ($text -match '(?<![0-9])8080(?![0-9])') { throw 'New-ArtemisConfig: the filled config names port 8080; another local service may serve there' }

  $bytes = $utf8.GetBytes($text)
  New-Item -ItemType Directory -Force (Split-Path $Out -Parent) | Out-Null
  [System.IO.File]::WriteAllBytes($Out, $bytes)
  return @{
    path = $Out
    sha256 = [System.Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
    template_sha256 = [System.Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($templateBytes)).ToLowerInvariant()
  }
}

# Starts `Exe serve` on 127.0.0.1:8090 against Model, with RunDir\artemis.toml, RunDir\logs, and its stderr in
# RunDir\stderr.txt; returns the Process once /health answers 200 and the 8090 listener is that process. The config file
# is empty (only the flags apply) unless -Config names a template: then it is New-ArtemisConfig's fill of that template
# with this call's model, key file, log directory and port. The flags still carry the same values. The Process gets a
# NoteProperty ArtemisConfig = @{ template; template_sha256; sha256; overrides } (the template and its hash are $null
# without -Config); overrides holds the port, the key file's path (never the key), the log directory, the model and
# -Extra. Refuses (before starting anything) when -Extra names port 8080 or sets a flag this function owns, when 8090 is
# busy, or when any `artemis` process runs. -WhatIf returns the argv without starting anything, and without filling
# the config.
function Start-ArtemisServer {
  [CmdletBinding(SupportsShouldProcess)]
  param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$RunDir,
    [Parameter(Mandatory)][string]$KeyFile,
    [string[]]$Extra = @(),
    [string]$Config
  )
  $Exe = Resolve-FullPath $Exe
  $Model = Resolve-FullPath $Model
  $RunDir = Resolve-FullPath $RunDir
  $KeyFile = Resolve-FullPath $KeyFile
  if ($Config) { $Config = Resolve-FullPath $Config }
  foreach ($e in $Extra) {
    if ($e -match '(?<![0-9])8080(?![0-9])') { throw "Start-ArtemisServer ($Name): an argument names port 8080 ('$e'); another local service may serve there" }
  }
  foreach ($e in $Extra) {
    if ($e -match '^--(config|model|host|port|api-key-file|log-dir|stop-on-stdin-close)(=|$)') {
      throw "Start-ArtemisServer ($Name): -Extra may not set $($Matches[0].TrimEnd('=')); this function owns it"
    }
  }
  if (Test-PortBusy $script:Port) {
    $owner = Get-PortOwner $script:Port
    $who = if ($owner.Count) { " (listener pid $($owner -join ', '))" } else { '' }
    throw "Start-ArtemisServer ($Name): port $($script:Port) is in use$who; refusing to start"
  }
  $running = @(Get-Process -Name artemis -ErrorAction SilentlyContinue)
  if ($running.Count) {
    throw "Start-ArtemisServer ($Name): an artemis process is running (pid $(($running | ForEach-Object Id) -join ', ')); refusing to start"
  }
  $needed = @(@('the executable', $Exe), @('the model', $Model), @('the key file', $KeyFile))
  if ($Config) { $needed += , @('the config template', $Config) }
  foreach ($f in $needed) {
    if (-not (Test-Path -LiteralPath $f[1] -PathType Leaf)) { throw "Start-ArtemisServer ($Name): $($f[0]) $($f[1]) does not exist" }
  }

  $configPath = Join-Path $RunDir 'artemis.toml'
  $logDir = Join-Path $RunDir 'logs'
  $stderr = Join-Path $RunDir 'stderr.txt'
  $argList = @('serve', '--config', $configPath, '--model', $Model, '--host', '127.0.0.1', '--port', "$($script:Port)",
    '--api-key-file', $KeyFile, '--log-dir', $logDir, '--stop-on-stdin-close') + $Extra
  if (-not $PSCmdlet.ShouldProcess("artemis serve ($Name)", 'start')) { return @($Exe) + $argList }

  New-Item -ItemType Directory -Force $logDir | Out-Null
  if ($Config) {
    $filled = New-ArtemisConfig -Template $Config -Out $configPath -Model $Model -KeyFile $KeyFile -LogDir $logDir -Port $script:Port
    $summary = @{ template = $Config; template_sha256 = $filled.template_sha256; sha256 = $filled.sha256 }
  } else {
    [System.IO.File]::WriteAllText($configPath, '')
    $summary = @{ template = $null; template_sha256 = $null; sha256 = [System.Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData([byte[]]@())).ToLowerInvariant() }
  }
  $summary.overrides = @{ port = $script:Port; api_key_file = $KeyFile; log_dir = $logDir; model = $Model; extra = @($Extra) }
  # A finally, not a catch: a stopped pipeline (Ctrl+C in the health wait) skips a catch and would orphan the server.
  # Start-Child is inside the try too, and the finally falls back on $script:Live (set inside Start-Child) for the
  # instant between Start-Child returning and $proc being assigned.
  $proc = $null
  $ok = $false
  try {
    $proc = Start-Child -Exe $Exe -ArgList $argList -StderrFile $stderr
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    while ($true) {
      if ($proc.HasExited) {
        $code = $proc.ExitCode
        Close-Drain $proc
        throw "Start-ArtemisServer ($Name): artemis serve exited with code $code before /health answered. stderr tail:`n$(Get-Tail $stderr)"
      }
      $h = Get-Health
      if ($h -and $h.Status -eq 200) { break }
      if ($clock.Elapsed.TotalSeconds -gt 60) {
        throw "Start-ArtemisServer ($Name): /health did not answer 200 within 60 s. stderr tail:`n$(Get-Tail $stderr)"
      }
      Start-Sleep -Milliseconds 250
    }
    $owner = Get-PortOwner $script:Port
    if ($owner.Count -ne 1 -or $owner[0] -ne $proc.Id) {
      throw "Start-ArtemisServer ($Name): port $($script:Port) is held by pid $($owner -join ', '), not by the server just started (pid $($proc.Id))"
    }
    Add-Member -InputObject $proc -NotePropertyName ArtemisConfig -NotePropertyValue $summary -Force
    $ok = $true
  } finally {
    if (-not $ok) {
      $victim = if ($proc) { $proc } else { $script:Live }
      if ($victim) { Stop-Child $victim }
    }
  }
  return $proc
}

# Stops a server Start-ArtemisServer started: closes its stdin (the stop), waits 90 s (serve's 30 s grace, finish's
# 20 s and the unload), then kills. Returns @{graceful; exit_code; seconds; exit_line; port_free}: graceful means it
# exited by itself within 90 s with code 0 and its log's last line is `exit` with reason `shutdown`.
function Stop-ArtemisServer {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][System.Diagnostics.Process]$Proc,
    [Parameter(Mandatory)][string]$RunDir
  )
  $RunDir = Resolve-FullPath $RunDir
  $clock = [System.Diagnostics.Stopwatch]::StartNew()
  try { $Proc.StandardInput.Close() } catch { }
  $exited = $Proc.WaitForExit(90000)
  if (-not $exited) {
    Write-Warning "artemis serve (pid $($Proc.Id)) did not exit within 90 s of its stdin closing; killing it"
    try { $Proc.Kill($true); $null = $Proc.WaitForExit(10000) } catch { }
  }
  $seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
  $code = if ($Proc.HasExited) { $Proc.ExitCode } else { $null }
  Close-Drain $Proc
  $Proc.Dispose()

  $exitLine = $false
  $last = Get-ChildItem -LiteralPath (Join-Path $RunDir 'logs') -Filter 'artemis-*.jsonl' -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -Last 1
  if ($last) {
    $line = Get-Content -LiteralPath $last.FullName -Tail 5 | Where-Object { $_.Trim() } | Select-Object -Last 1
    try { $j = $line | ConvertFrom-Json; $exitLine = ($j.event -eq 'exit' -and $j.reason -eq 'shutdown') } catch { }
  }
  # The listener is gone the moment the process is, but give the table a moment.
  $free = $false
  for ($i = 0; $i -lt 20 -and -not $free; $i++) {
    $free = -not (Test-PortBusy $script:Port)
    if (-not $free) { Start-Sleep -Milliseconds 500 }
  }
  if (-not $free) { Write-Warning "port $($script:Port) is still busy after the server stopped" }
  $graceful = ($exited -and $code -eq 0 -and $exitLine)
  if ($exited -and $code -eq 0 -and -not $exitLine) { Write-Warning "artemis serve exited 0 but its log's last line is not exit/shutdown" }
  return @{ graceful = $graceful; exit_code = $code; seconds = $seconds; exit_line = $exitLine; port_free = $free }
}

# Throws if the key in KeyFile is in any file under Paths (a path may be a directory or a single file), except the
# files in -Expected, which must hold it (so a moved config can't hide a leak). Paths compare in one canonical spelling
# (8.3 short names expanded). A path that does not exist or can't be read throws. Names the file, never the key. Returns
# how many files it scanned.
function Assert-NoKey {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$KeyFile,
    [Parameter(Mandatory)][string[]]$Paths,
    [AllowNull()][string[]]$Expected = @()
  )
  # A $null (an empty array that went through an `if` expression) is no entry; a blank entry would resolve to the current
  # directory, so it is an error, not a path.
  $Expected = @($Expected | Where-Object { $null -ne $_ })
  if (@($Expected | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count) { throw 'Assert-NoKey: -Expected holds an empty path' }
  $key = Read-Key (Resolve-FullPath $KeyFile)
  $expectedFull = @($Expected | ForEach-Object { Get-CanonicalPath $_ })
  $scanned = 0
  $leaks = @()
  $seen = @{}
  foreach ($p in $Paths) {
    $root = Get-CanonicalPath $p
    # A root that is missing or unreadable is a failure, not a clean scan; so is any directory the walk can't read.
    if (-not (Test-Path -LiteralPath $root)) { throw "Assert-NoKey: $root does not exist, so it was not scanned" }
    # A path may be a directory (everything under it) or a single file (the evidence file).
    $files = if (Test-Path -LiteralPath $root -PathType Leaf) { @(Get-Item -LiteralPath $root -Force -ErrorAction Stop) } else { @(Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction Stop) }
    foreach ($f in $files) {
      $name = Get-CanonicalPath $f.FullName
      if ($seen.ContainsKey($name.ToLowerInvariant())) { continue }
      $seen[$name.ToLowerInvariant()] = $true
      $scanned++
      if ($expectedFull -contains $name) { continue }
      if (Test-FileHolds $name $key) { $leaks += $name }
    }
  }
  $missing = @($expectedFull | Where-Object { -not ((Test-Path -LiteralPath $_ -PathType Leaf) -and (Test-FileHolds $_ $key)) })
  if ($leaks.Count) { throw "the key is in $($leaks.Count) file(s) it must not be in: $($leaks -join '; ')" }
  if ($missing.Count) { throw "expected file(s) that should hold the key do not (moved or empty?): $($missing -join '; ')" }
  return $scanned
}

# The child Start-ArtemisServer started and Stop-ArtemisServer has not stopped yet, or $null. For a caller whose own
# variable was never assigned because the pipeline was stopped as Start-ArtemisServer returned.
function Get-ArtemisServerProcess {
  [CmdletBinding()]
  param()
  # HasExited reads $null on a disposed Process in PowerShell, so a stale handle is not "running": test it for $false.
  if ($script:Live -and ($script:Live.HasExited -eq $false)) { return $script:Live }
  return $null
}

Export-ModuleMember -Function New-ArtemisKey, New-ArtemisConfig, Start-ArtemisServer, Stop-ArtemisServer, Assert-NoKey, Assert-Preflight, Get-ArtemisServerProcess
