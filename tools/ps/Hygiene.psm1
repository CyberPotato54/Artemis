#Requires -Version 7.2
# Speed hygiene for tools\run-gpu-tests.ps1: the CPU affinity helpers and the PCIe link check.
#   Get-ArtemisAllCpuMask   the system's affinity mask: every CPU in the processor group, whatever this process is
#                           pinned to
#   Get-ArtemisAffinity     the current process's affinity mask
#   Set-ArtemisAffinity     sets the current process's mask (children inherit it) and returns the mask read back
#   Get-ArtemisProcessAffinity  another process's mask, read fresh (Process caches it)
#   ConvertTo-ArtemisMaskLong  a UInt64 mask's bits as a long (no throw above Int64.MaxValue)
#   Format-ArtemisMask      0x and lower-case hex
#   Test-ArtemisLink        runs `gpu_link.py check` and returns @{ adapter; pcie_gen; ok; why }
#
# "All CPUs" is the system mask from kernel32's GetProcessAffinityMask, never derived from
# [Environment]::ProcessorCount: .NET 6 and later report the CPUs the process may run on, so a console pinned to one
# CCD (by Process Lasso, say) reports half the machine and `(1 -shl ProcessorCount) - 1` would name the wrong CPUs.

if (-not ('ArtemisHygieneNative' -as [type])) {
  Add-Type -Namespace '' -Name 'ArtemisHygieneNative' -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetProcessAffinityMask(System.IntPtr process, out System.UIntPtr processMask,
  out System.UIntPtr systemMask);
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern System.IntPtr GetCurrentProcess();
'@
}

# A mask's bits as a [long]: a [long] cast of a UInt64 above Int64.MaxValue throws (a full 64-CPU group mask is
# 0xFFFFFFFFFFFFFFFF), so the bits are reinterpreted and the top bit becomes the sign.
function ConvertTo-ArtemisMaskLong {
  [OutputType([long])]
  param([Parameter(Mandatory)][uint64]$Bits)
  return [BitConverter]::ToInt64([BitConverter]::GetBytes($Bits), 0)
}

function Get-ArtemisMasks {
  $proc = [UIntPtr]::Zero
  $sys = [UIntPtr]::Zero
  if (-not [ArtemisHygieneNative]::GetProcessAffinityMask([ArtemisHygieneNative]::GetCurrentProcess(), [ref]$proc, [ref]$sys)) {
    throw "GetProcessAffinityMask failed (Win32 error $([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
  }
  return @{ process = (ConvertTo-ArtemisMaskLong $proc.ToUInt64()); system = (ConvertTo-ArtemisMaskLong $sys.ToUInt64()) }
}

function Get-ArtemisAllCpuMask {
  [OutputType([long])]
  param()
  return [long](Get-ArtemisMasks).system
}

function Get-ArtemisAffinity {
  [OutputType([long])]
  param()
  return [long](Get-ArtemisMasks).process
}

function Set-ArtemisAffinity {
  [OutputType([long])]
  param([Parameter(Mandatory)][long]$Mask)
  if ($Mask -eq 0) { throw 'an affinity mask of 0 names no CPU' }
  $p = [System.Diagnostics.Process]::GetCurrentProcess()
  $p.ProcessorAffinity = [IntPtr]$Mask
  $back = Get-ArtemisAffinity
  if ($back -ne $Mask) {
    throw "set the affinity mask to $(Format-ArtemisMask $Mask), but it reads back as $(Format-ArtemisMask $back)"
  }
  return $back
}

# The mask of another process (a server the runner started), read fresh: System.Diagnostics.Process caches
# ProcessorAffinity after the first get, so a re-pin made since (Process Lasso, say) shows only after Refresh().
# $null when it can't be read (the process is gone).
function Get-ArtemisProcessAffinity {
  [OutputType([long])]
  param([Parameter(Mandatory)][System.Diagnostics.Process]$Process)
  try {
    $Process.Refresh()
    if ($Process.HasExited) { return $null }
    return [long]$Process.ProcessorAffinity
  } catch { return $null }
}

function Format-ArtemisMask {
  [OutputType([string])]
  param([Parameter(Mandatory)]$Mask)
  # A PowerShell hex literal that fits 32 bits is an Int32 (0xFFFFFFFF is -1): read an [int] as its unsigned bits. A
  # [long] is taken as it is, so a real 64-bit mask such as 0xFFFFFFFF80000000 keeps its upper half.
  if ($Mask -is [int] -and $Mask -lt 0) { return '0x{0:x}' -f ([long]$Mask -band 0xFFFFFFFFL) }
  return '0x{0:x}' -f [long]$Mask
}

# CPU quietness. Background scanners (Windows Defender's MsMpEng and the search indexer's SearchIndexer) scanning new
# build outputs slowed llama.cpp's CPU expert threads about 2.3 times on 2026-11-01 while Artemis was unaffected. A
# speed block records the scanners' CPU seconds and the system's busy percentage at its start, and the scanners' CPU
# seconds again at its end.
#   Get-ArtemisScannerCpu     @{ MsMpEng = <seconds>; SearchIndexer = <seconds> }: the total processor time of each
#                             (summed over its instances), $null for one that is not running or can't be read
#   Get-ArtemisScannerDelta   the second reading less the first, per scanner; $null where either is missing or the
#                             scanner restarted in between (the time went down)
#   Get-ArtemisBusyPct        the whole system's busy percentage over a short sample, $null when it can't be read
function Get-ArtemisScannerCpu {
  $r = @{}
  foreach ($name in 'MsMpEng', 'SearchIndexer') {
    try {
      $seconds = @(Get-Process -Name $name -ErrorAction Stop) | ForEach-Object { $_.TotalProcessorTime.TotalSeconds }
      $r[$name] = [math]::Round(($seconds | Measure-Object -Sum).Sum, 2)
    } catch { $r[$name] = $null }
  }
  return $r
}

function Get-ArtemisScannerDelta {
  param($Before, $After)
  $r = @{}
  foreach ($name in 'MsMpEng', 'SearchIndexer') {
    $b = $(if ($Before) { $Before[$name] } else { $null })
    $a = $(if ($After) { $After[$name] } else { $null })
    if ($null -ne $b -and $null -ne $a -and $a -ge $b) { $r[$name] = [math]::Round($a - $b, 2) } else { $r[$name] = $null }
  }
  return $r
}

function Get-ArtemisBusyPct {
  param([int]$Seconds = 2)
  try {
    $c = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval $Seconds -MaxSamples 1 -ErrorAction Stop
    return [math]::Round($c.CounterSamples[0].CookedValue, 1)
  } catch { return $null }
}

function Test-ArtemisLink {
  param(
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$Repo,
    [string]$Name = '9070',
    [int]$MinGen = 4
  )
  $script = Join-Path $Repo 'tools\py\gpu_link.py'
  $text = ''
  try {
    $text = (& $Python -I $script check --name $Name --min-gen $MinGen 2>&1 | Out-String)
  } catch {
    return @{ adapter = $null; pcie_gen = $null; ok = $false; why = "gpu_link.py check could not run: $($_.Exception.Message)" }
  }
  $line = @($text -split "`r?`n" | Where-Object { $_.TrimStart().StartsWith('{') }) | Select-Object -Last 1
  $j = $null
  if ($line) { try { $j = $line | ConvertFrom-Json -ErrorAction Stop } catch { $j = $null } }
  if (-not $j) {
    return @{ adapter = $null; pcie_gen = $null; ok = $false; why = "gpu_link.py check printed no JSON line: $($text.Trim())" }
  }
  return @{ adapter = $j.adapter; pcie_gen = $j.pcie_gen; ok = ($j.ok -eq $true); why = $j.why }
}

Export-ModuleMember -Function Get-ArtemisScannerCpu, Get-ArtemisScannerDelta, Get-ArtemisBusyPct,
  ConvertTo-ArtemisMaskLong, Get-ArtemisAllCpuMask, Get-ArtemisAffinity, Set-ArtemisAffinity,
  Get-ArtemisProcessAffinity, Format-ArtemisMask, Test-ArtemisLink
