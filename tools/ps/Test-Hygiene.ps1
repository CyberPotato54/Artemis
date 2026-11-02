#Requires -Version 7.2
# CPU check of tools\ps\Hygiene.psm1: the mask helpers and Test-ArtemisLink with stub interpreters. It reads and sets
# this process's affinity (the widening test restores it in a finally), starts no artemis and touches no GPU. Exit 0
# when the checks pass, 1 when one fails.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force

$failures = @()
function Expect([bool]$Ok, [string]$What) {
  if ($Ok) { Write-Host "ok   $What" } else { Write-Host "FAIL $What"; $script:failures += $What }
}
function BitCount([long]$Mask) {
  $n = 0
  for ($b = 0; $b -lt 64; $b++) { if (($Mask -shr $b) -band 1) { $n++ } }
  return $n
}

# Format-ArtemisMask
Expect ((Format-ArtemisMask 0xFFFFFFFF) -ceq '0xffffffff') 'Format-ArtemisMask 0xFFFFFFFF is 0xffffffff'
Expect ((Format-ArtemisMask 0xFFFF0000) -ceq '0xffff0000') 'Format-ArtemisMask 0xFFFF0000 is 0xffff0000'
Expect ((Format-ArtemisMask 255) -ceq '0xff') 'Format-ArtemisMask 255 is 0xff'
Expect ((Format-ArtemisMask ([long]0xFFFFFFFF80000000)) -ceq '0xffffffff80000000') 'Format-ArtemisMask keeps the upper half of a real 64-bit [long] mask'

# ConvertTo-ArtemisMaskLong reinterprets the bits: a full 64-CPU group mask is 0xFFFFFFFFFFFFFFFF, which a [long] cast
# of the UInt64 refuses.
Expect ((ConvertTo-ArtemisMaskLong ([uint64]::MaxValue)) -eq -1L) 'ConvertTo-ArtemisMaskLong of a full 64-bit mask is -1 (no throw)'
Expect ((ConvertTo-ArtemisMaskLong ([uint64]9223372036854775808)) -eq [long]::MinValue) 'ConvertTo-ArtemisMaskLong keeps the top bit as the sign'
Expect ((ConvertTo-ArtemisMaskLong ([uint64]4294901760)) -eq 4294901760L) 'ConvertTo-ArtemisMaskLong leaves a mask below 2^63 as it is'
Expect ((Format-ArtemisMask (ConvertTo-ArtemisMaskLong ([uint64]::MaxValue))) -ceq '0xffffffffffffffff') 'a full 64-bit mask formats as 0xffffffffffffffff'
Expect ((BitCount (ConvertTo-ArtemisMaskLong ([uint64]::MaxValue))) -eq 64) 'a full 64-bit mask has 64 bits set'

# Get-ArtemisAllCpuMask is the system mask, so its bit count is the machine's logical processors, whatever this console
# is pinned to.
$all = Get-ArtemisAllCpuMask
$logical = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Expect ($all -is [long]) 'Get-ArtemisAllCpuMask returns a long'
Expect ($all -ne 0) 'Get-ArtemisAllCpuMask is not 0'
Expect ((BitCount $all) -eq $logical) "Get-ArtemisAllCpuMask has $logical bits set (the logical processors); it is $(Format-ArtemisMask $all)"

# Set-ArtemisAffinity: the current mask comes back, and the mask can be widened to the whole system and read back.
$orig = Get-ArtemisAffinity
try {
  Expect ((Set-ArtemisAffinity -Mask $orig) -eq $orig) 'Set-ArtemisAffinity with the current mask returns it'
  $back = Set-ArtemisAffinity -Mask $all
  Expect ($back -eq $all) "Set-ArtemisAffinity widens to the system mask ($(Format-ArtemisMask $all)) and reads it back"
  Expect ((Get-ArtemisAffinity) -eq $all) 'Get-ArtemisAffinity agrees after the widening'
  $threw = $false
  try { Set-ArtemisAffinity -Mask 0 | Out-Null } catch { $threw = $true }
  Expect $threw 'Set-ArtemisAffinity refuses a mask of 0'
} finally {
  Set-ArtemisAffinity -Mask $orig | Out-Null
}
Expect ((Get-ArtemisAffinity) -eq $orig) "the original mask $(Format-ArtemisMask $orig) is restored"

# Get-ArtemisProcessAffinity sees a re-pin made through another handle (Process caches ProcessorAffinity until Refresh).
$child = Start-Process -FilePath ping.exe -ArgumentList '-n', '30', '127.0.0.1' -PassThru -WindowStyle Hidden
try {
  $before = Get-ArtemisProcessAffinity -Process $child
  Expect ($before -eq (Get-ArtemisAffinity)) 'a child starts with the mask it inherited'
  $low = $before -band (-$before)
  $other = [System.Diagnostics.Process]::GetProcessById($child.Id)
  $other.ProcessorAffinity = [IntPtr]$low
  Expect ((Get-ArtemisProcessAffinity -Process $child) -eq $low) "Get-ArtemisProcessAffinity sees the child re-pinned to $(Format-ArtemisMask $low) through a second handle"
} finally {
  Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
  $child.WaitForExit(10000) | Out-Null
}
$gone = Get-ArtemisProcessAffinity -Process $child
Expect ($null -eq $gone) 'Get-ArtemisProcessAffinity is $null for a process that has exited'

# Test-ArtemisLink with stub interpreters: a .cmd that ignores its arguments and prints a canned JSON line.
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
  function Stub([string]$Name, [string]$Line, [int]$Code) {
    $path = Join-Path $tmp "$Name.cmd"
    Set-Content -LiteralPath $path -Value "@echo off`r`necho $Line`r`nexit /b $Code" -Encoding ascii
    return $path
  }
  $slow = Stub 'slow' '{"adapter": "AMD Radeon RX 9070 XT", "pcie_gen": 3, "ok": false, "why": "gen 3"}' 1
  $good = Stub 'good' '{"adapter": "AMD Radeon RX 9070 XT", "pcie_gen": 4, "ok": true, "why": "PCIe gen 4"}' 0
  $junk = Stub 'junk' 'no json here' 2
  $r = Test-ArtemisLink -Python $slow -Repo $tmp
  Expect ($r.ok -eq $false -and $r.pcie_gen -eq 3) 'Test-ArtemisLink on a gen 3 stub returns ok = false, pcie_gen 3'
  $r = Test-ArtemisLink -Python $good -Repo $tmp
  Expect ($r.ok -eq $true -and $r.pcie_gen -eq 4 -and $r.adapter -eq 'AMD Radeon RX 9070 XT') 'Test-ArtemisLink on a gen 4 stub returns ok = true, pcie_gen 4'
  $r = Test-ArtemisLink -Python $junk -Repo $tmp
  Expect ($r.ok -eq $false -and $null -eq $r.pcie_gen) 'Test-ArtemisLink on a stub that prints no JSON returns ok = false'
  $r = Test-ArtemisLink -Python (Join-Path $tmp 'missing.exe') -Repo $tmp
  Expect ($r.ok -eq $false) 'Test-ArtemisLink with a missing interpreter returns ok = false'
} finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

# The runner's own functions (SpeedHygiene, LinkAfter, WithLink, AffinityFields, MaskText, LlamaChoice, MeasureLlama,
# MeasureLlamaSearch, SamePositions, AttnDepthJudge, RunMeasured, MaskNote, AddProblem), taken out of the script by the
# parser and run against stub interpreters and a Record that collects.
$runner = Join-Path (Split-Path $PSScriptRoot -Parent) 'run-gpu-tests.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($runner, [ref]$null, [ref]$null)
foreach ($fn in 'SpeedHygiene', 'LinkAfter', 'WithLink', 'AffinityFields', 'MaskText', 'LlamaChoice', 'MeasureLlama', 'MeasureLlamaSearch', 'SamePositions', 'AttnDepthJudge', 'RunMeasured', 'MaskNote', 'AddProblem') {
  $def = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $fn }, $true)
  if (-not $def) { Expect $false "run-gpu-tests.ps1 defines $fn"; continue }
  Invoke-Expression $def.Extent.Text
}
$tmp2 = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-run-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp2 | Out-Null
try {
  function Stub2([string]$Name, [string]$Line, [int]$Code) {
    $path = Join-Path $tmp2 "$Name.cmd"
    Set-Content -LiteralPath $path -Value "@echo off`r`necho $Line`r`nexit /b $Code" -Encoding ascii
    return $path
  }
  $g4 = Stub2 'g4' '{"adapter": "A", "pcie_gen": 4, "ok": true, "why": "PCIe gen 4"}' 0
  $g3 = Stub2 'g3' '{"adapter": "A", "pcie_gen": 3, "ok": false, "why": "gen 3"}' 1
  $repo = $tmp2
  $GpuName = 'A'
  $runnerMask = Get-ArtemisAffinity
  $script:records = @()
  $script:failed = @()
  $script:hygieneNow = $null
  $script:linkGenBefore = $null
  $script:hygieneBlock = $null
  function Record([hashtable]$r) {
    if ($script:hygieneNow -and -not $r.ContainsKey('hygiene')) { $r.hygiene = $script:hygieneNow }
    $script:records += $r
  }
  function DesktopGpuMib { return 123.4 }
  # The scanners' CPU seconds and the busy percentage, as the runner's wrappers would read them.
  $script:scanNow = @{ MsMpEng = 100.0; SearchIndexer = 10.0 }
  function ScannerCpu { return $script:scanNow.Clone() }
  function BusyPct { return 3.2 }

  # A good block: gen 4 before and after.
  $python = $g4
  $h = SpeedHygiene 'blk'
  Expect ($h.pcie_gen -eq 4 -and $script:records[-1].status -eq 'pass') 'SpeedHygiene at gen 4 records a pass'
  Record @{ test = 'mid' }
  Expect ($script:records[-1].hygiene.pcie_gen -eq 4) 'a record between SpeedHygiene and LinkAfter carries hygiene'
  $link = LinkAfter 'blk'
  Expect ($link.link_dropped -eq $false -and $script:records[-1].test -eq 'blk link after' -and $script:records[-1].status -eq 'pass') 'LinkAfter at gen 4 passes'
  $agg = WithLink @{ test = 'agg'; status = 'pass'; problems = @() } $link
  Expect ($agg.status -eq 'pass' -and $agg.link_dropped -eq $false -and $agg.hygiene.pcie_gen -eq 4 -and $script:failed.Count -eq 0) 'WithLink on a good link keeps the aggregate a pass and adds hygiene'
  Expect ($null -eq (LinkAfter 'blk')) 'LinkAfter on a closed block does nothing'

  # The CPU-quietness record: the scanners' seconds and the busy percentage at the start, the deltas at the end.
  $script:scanNow = @{ MsMpEng = 100.0; SearchIndexer = 10.0 }
  $h = SpeedHygiene 'quiet'
  Expect ($h.scanner_cpu_s.MsMpEng -eq 100.0 -and $h.scanner_cpu_s.SearchIndexer -eq 10.0 -and $h.busy_pct -eq 3.2) 'SpeedHygiene records the scanners CPU seconds and the busy percentage'
  Expect ($script:records[-1].hygiene.scanner_cpu_s.MsMpEng -eq 100.0 -and $script:records[-1].hygiene.busy_pct -eq 3.2) 'the hygiene record carries them'
  $script:scanNow = @{ MsMpEng = 112.5; SearchIndexer = 10.25 }
  $link = LinkAfter 'quiet'
  Expect ($link.scanner_cpu_delta_s.MsMpEng -eq 12.5 -and $link.scanner_cpu_delta_s.SearchIndexer -eq 0.25) 'LinkAfter returns the scanners CPU seconds used during the block'
  Expect ($script:records[-1].test -eq 'quiet link after' -and $script:records[-1].scanner_cpu_delta_s.MsMpEng -eq 12.5) 'the link-after record carries the deltas'
  $agg = WithLink @{ test = 'agg'; status = 'pass' } $link
  Expect ($agg.scanner_cpu_delta_s.MsMpEng -eq 12.5 -and $agg.hygiene.busy_pct -eq 3.2) 'WithLink folds the deltas and the start reading into the aggregate'
  $script:scanNow = @{ MsMpEng = 100.0; SearchIndexer = 10.0 }

  # Get-ArtemisScannerDelta: a missing reading or a scanner that restarted (its time went down) is $null.
  $d = Get-ArtemisScannerDelta @{ MsMpEng = 10.0; SearchIndexer = $null } @{ MsMpEng = 40.5; SearchIndexer = 5.0 }
  Expect ($d.MsMpEng -eq 30.5 -and $null -eq $d.SearchIndexer) 'Get-ArtemisScannerDelta subtracts, and gives $null for a missing reading'
  $d = Get-ArtemisScannerDelta @{ MsMpEng = 50.0; SearchIndexer = 1.0 } @{ MsMpEng = 5.0; SearchIndexer = 1.0 }
  Expect ($null -eq $d.MsMpEng -and $d.SearchIndexer -eq 0.0) 'Get-ArtemisScannerDelta gives $null for a scanner whose time went down'
  $d = Get-ArtemisScannerDelta $null $null
  Expect ($null -eq $d.MsMpEng -and $null -eq $d.SearchIndexer) 'Get-ArtemisScannerDelta of two missing readings is all $null'
  $live = Get-ArtemisScannerCpu
  Expect ($live.ContainsKey('MsMpEng') -and $live.ContainsKey('SearchIndexer')) 'Get-ArtemisScannerCpu names both scanners, whether or not they run'

  # AddProblem keeps the first problem and adds the next.
  $pr = @{}
  AddProblem $pr 'affinity'
  Expect ($pr.problem -eq 'affinity') 'AddProblem sets a first problem'
  AddProblem $pr 'link dropped'
  Expect ($pr.problem -eq 'affinity, link dropped') 'AddProblem keeps the first problem and adds the second'
  AddProblem $pr 'affinity'
  Expect ($pr.problem -eq 'affinity, link dropped') 'AddProblem does not add a name twice'
  $pr = @{ test = 'x'; status = 'fail'; problem = 'affinity'; problems = @() }
  $dl = @{ link_dropped = $true; pcie_gen = 3; why = 'gen 3'; hygiene = @{}; scanner_cpu_delta_s = $null }
  $pr = WithLink $pr $dl
  Expect ($pr.problem -eq 'affinity, link dropped') 'WithLink keeps an affinity problem beside link dropped'

  # MaskNote: a run on the runner's mask adds nothing; a moved one adds "<tag>: <why>" and its fields.
  $mk = @{}
  $pl2 = [System.Collections.Generic.List[string]]::new()
  MaskNote @{ exit = 0; affinity = 0xffff0000L; affinity_at_stop = 0xffff0000L; affinity_problem = $null } 'a' $mk $pl2
  MaskNote @{ exit = 0; affinity = 0xffff0000L; affinity_at_stop = 0x3L; affinity_problem = 'moved' } 'b' $mk $pl2
  Expect ($pl2.Count -eq 1 -and $pl2[0] -eq 'b: moved' -and $mk['a'].affinity -ceq '0xffff0000' -and -not $mk['a'].ContainsKey('status') -and $mk['b'].status -eq 'fail' -and $mk['b'].problem -eq 'affinity') 'MaskNote files the masks under their tags and collects the problems'
  Record @{ test = 'after' }
  Expect (-not $script:records[-1].ContainsKey('hygiene')) 'a record after LinkAfter carries no hygiene'

  # A dropped link: gen 4 at the start, gen 3 at the end.
  $python = $g4
  SpeedHygiene 'drop' | Out-Null
  $python = $g3
  $link = LinkAfter 'drop'
  Expect ($link.link_dropped -eq $true -and $script:records[-1].problem -eq 'link dropped' -and $script:failed -contains 'drop link dropped') 'LinkAfter after a fall to gen 3 records link dropped and fails the block'
  $agg = WithLink @{ test = 'agg'; status = 'pass'; problems = @('x') } $link
  Expect ($agg.status -eq 'fail' -and $agg.problem -eq 'link dropped' -and $agg.link_dropped -eq $true -and @($agg.problems).Count -eq 2) 'WithLink fails the aggregate with problem link dropped'

  # A gen 3 start: the hygiene record fails and SpeedHygiene throws, leaving no block open.
  $python = $g3
  $threw = $false
  try { SpeedHygiene 'bad' | Out-Null } catch { $threw = $true }
  Expect ($threw -and $script:records[-1].status -eq 'fail' -and $script:failed -contains 'bad hygiene' -and $null -eq $script:hygieneNow) 'SpeedHygiene at gen 3 records a fail, throws and opens no block'

  # The runner's own mask changed since it set it.
  $python = $g4
  $runnerMask = 0x7
  $threw = $false
  try { SpeedHygiene 'mask' | Out-Null } catch { $threw = $true }
  Expect ($threw -and $script:records[-1].problem -match 'affinity') 'SpeedHygiene fails when the runner no longer holds the mask it set'
  $runnerMask = Get-ArtemisAffinity

  # AffinityFields: a server whose mask at the stop differs from the start.
  $f = AffinityFields @{ affinity = 0xffff0000L; affinity_at_stop = 0x3L; affinity_problem = 'moved' }
  Expect ($f.status -eq 'fail' -and $f.problem -eq 'affinity' -and $f.affinity -ceq '0xffff0000' -and $f.affinity_at_stop -ceq '0x3') 'AffinityFields fails a record whose mask moved'
  $f = AffinityFields @{ affinity = 0xffff0000L; affinity_at_stop = 0xffff0000L; affinity_problem = $null }
  Expect (-not $f.ContainsKey('status') -and $f.affinity_at_stop -ceq '0xffff0000') 'AffinityFields leaves a record whose mask held alone'

  # RunMeasured against short-lived pwsh children (no GPU): the mask is read at the start and until the exit.
  $pw = (Get-Command pwsh).Source
  $o = Join-Path $tmp2 'rm.out'; $e = Join-Path $tmp2 'rm.err'
  $runnerMask = Get-ArtemisAffinity
  $r = RunMeasured $pw @('-NoProfile', '-Command', 'Start-Sleep 1; Write-Output ''a b''') $o $e
  Expect ($r.exit -eq 0 -and $r.affinity -eq $runnerMask -and $r.affinity_at_stop -eq $runnerMask -and -not $r.affinity_problem) 'RunMeasured: a child on the runner mask reads it at the start and at the stop, with no problem'
  Expect ((Get-Content -LiteralPath $o -Raw).Trim() -ceq 'a b') 'RunMeasured: an argument with a space reaches the child whole and its output is redirected'
  $r = RunMeasured $pw @('-NoProfile', '-Command', 'Start-Sleep 1; exit 3') $o $e
  Expect ($r.exit -eq 3) 'RunMeasured returns the child exit code'
  $low = $runnerMask -band (-$runnerMask)
  $runnerMask = $low -bxor $runnerMask
  if ($runnerMask -eq 0) { $runnerMask = $low -shl 1 }
  $r = RunMeasured $pw @('-NoProfile', '-Command', 'Start-Sleep 1') $o $e
  Expect ($r.exit -eq 0 -and $r.affinity_problem -like '*not the runner*') 'RunMeasured sets affinity_problem when the child mask is not the runner mask'
  $runnerMask = Get-ArtemisAffinity
  # A child re-pinned after it started: the stop read shows it.
  $helper = Join-Path $tmp2 'repin.ps1'
  Set-Content -LiteralPath $helper -Encoding utf8 -Value 'Start-Sleep 1; (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]([long]$args[0]); Start-Sleep 2'
  $r = RunMeasured $pw @('-NoProfile', '-File', $helper, "$low") $o $e
  Expect ($r.affinity -eq $runnerMask -and $r.affinity_at_stop -eq $low -and $r.affinity_problem) 'RunMeasured sees a child re-pinned mid-run (the last read differs and is a problem)'
} finally {
  Remove-Item -LiteralPath $tmp2 -Recurse -Force -ErrorAction SilentlyContinue
}

# LlamaChoice and MeasureLlama against a stub interpreter that prints a console line, as the real scripts do, and
# copies a canned result to the --out path.
$tmp3 = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-llama-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp3 | Out-Null
try {
  $stubPy = Join-Path $tmp3 'py.cmd'
  Set-Content -LiteralPath $stubPy -Encoding ascii -Value "@echo off`r`necho stub progress line`r`ncopy /y `"%~dp0canned.json`" %7 >nul`r`nexit /b %STUB_EXIT%"
  $script:cannedPath = Join-Path $tmp3 'canned.json'
  function Canned([string]$Affinity, [string]$AtStop, [string]$Error = '', [string]$Asked = '') {
    $o = [ordered]@{ config = 'c'; device = @{ id = 'Vulkan1'; name = 'X' }; args = @()
      affinity = $(if ($Affinity) { $Affinity } else { $null })
      affinity_at_stop = $(if ($AtStop) { $AtStop } else { $null }); runs = @(1); median = 90.5 }
    if ($Asked) { $o.affinity_asked = $Asked }
    if ($Error) { $o.error = $Error; $o.Remove('median') }
    Set-Content -LiteralPath $script:cannedPath -Value ($o | ConvertTo-Json -Depth 4) -Encoding utf8
  }
  $python = $stubPy
  $repo = $tmp3; $Model = 'm'; $Oracle = $tmp3; $GpuName = 'A'
  $runnerMask = [Convert]::ToInt64('ffff0000', 16)
  $env:STUB_EXIT = '0'
  $script:llamaSearchBest = $null
  $LlamaConfig = ''
  $searched = Join-Path $tmp3 'searched.json'
  Set-Content -LiteralPath $searched -Value '{"best":{"affinity":"0xffff"},"finalists":[],"driver":"d","llama_build":"b"}' -Encoding utf8
  $searchedInt = Join-Path $tmp3 'searched-int.json'
  Set-Content -LiteralPath $searchedInt -Value '{"best":{"affinity":4294901760},"finalists":[],"driver":"d","llama_build":"b"}' -Encoding utf8
  $notSearched = Join-Path $tmp3 'plain.json'
  Set-Content -LiteralPath $notSearched -Value '{"best":{}}' -Encoding utf8

  # LlamaChoice: search beats file; a file needs best, finalists, driver and llama_build; a missing file throws.
  Expect ((LlamaChoice).source -eq 'plan 2a') 'LlamaChoice with no search and no file is plan 2a'
  $LlamaConfig = $searched
  Expect ((LlamaChoice).source -eq 'file') 'LlamaChoice takes a searched -LlamaConfig as file'
  $script:llamaSearchBest = $searchedInt
  Expect ((LlamaChoice).source -eq 'search') "LlamaChoice prefers the search block's best.json over -LlamaConfig"
  $script:llamaSearchBest = $null
  $LlamaConfig = $notSearched
  Expect ((LlamaChoice 3>$null).source -eq 'plan 2a') 'LlamaChoice falls back to plan 2a for a file that is not a searched result'
  $LlamaConfig = Join-Path $tmp3 'missing.json'
  $threw = $false
  try { LlamaChoice | Out-Null } catch { $threw = $true }
  Expect $threw 'LlamaChoice throws for a -LlamaConfig that is not a file'
  $LlamaConfig = ''

  $plan = @{ source = 'plan 2a'; file = $null }
  Canned '0xffff0000' '0xffff0000'
  $r = MeasureLlama (Join-Path $tmp3 'd1') $plan 0 $false
  Expect ($r -is [hashtable]) "MeasureLlama returns one record; the script's console lines don't join it"
  Expect ($r.status -eq 'ran' -and $r.median -eq 90.5 -and $r.config_source -eq 'plan 2a' -and $r.affinity_asked -ceq '0xffff0000') "MeasureLlama on plan 2a at the runner's mask is a ran record asking the runner's mask"
  Canned '0xffff' '0xffff'
  $r = MeasureLlama (Join-Path $tmp3 'd2') $plan 0 $false
  Expect ($r.status -eq 'fail' -and $r.problem -eq 'affinity') "MeasureLlama fails plan 2a when the server is not on the runner's mask"

  $srch = @{ source = 'search'; file = $searched }
  Canned '0xffff' '0xffff' '' '0xffff'
  $r = MeasureLlama (Join-Path $tmp3 'd3') $srch 0 $false
  Expect ($r.status -eq 'ran' -and $r.affinity_asked -ceq '0xffff') "MeasureLlama accepts a searched mask that differs from the runner's"
  Canned '0xffff0000' '0xffff0000' '' '0xffff'
  $r = MeasureLlama (Join-Path $tmp3 'd4') $srch 0 $false
  Expect ($r.status -eq 'fail' -and $r.problem -eq 'affinity') "MeasureLlama fails a searched run whose server is on the runner's mask instead of the asked one"
  Canned '0xffff' '0xff' '' '0xffff'
  $r = MeasureLlama (Join-Path $tmp3 'd5') $srch 0 $false
  Expect ($r.status -eq 'fail' -and $r.problem -eq 'affinity') 'MeasureLlama fails a mask that drifted before the stop'
  # An integer mask in best.json, with and without the script reporting the mask it asked for.
  $srchInt = @{ source = 'search'; file = $searchedInt }
  Canned '0xffff0000' '0xffff0000'
  $r = MeasureLlama (Join-Path $tmp3 'd6') $srchInt 0 $false
  Expect ($r.status -eq 'ran' -and $r.affinity_asked -ceq '0xffff0000') 'MeasureLlama reads an integer best.affinity as the number it is'
  Canned '0xffff0000' '0xffff0000' '' '0xffff0000'
  $r = MeasureLlama (Join-Path $tmp3 'd7') $srchInt 0 $false
  Expect ($r.status -eq 'ran' -and $r.affinity_asked -ceq '0xffff0000') 'MeasureLlama uses the mask the script reports it asked for'

  # A server that never became ready: a failure at the default context, the figure at 65k; other errors fail at both.
  $env:STUB_EXIT = '1'
  Canned '' '' 'llama-server exited with 1; see x'
  $r = MeasureLlama (Join-Path $tmp3 'd8') $plan 0 $false
  Expect ($r.status -eq 'fail' -and $r.problem -like 'llama-server exited*') 'MeasureLlama fails a load failure at the default context'
  $r = MeasureLlama (Join-Path $tmp3 'd9') $plan 65536 $true
  Expect ($r.status -eq 'ran' -and $r.error -like 'llama-server exited*' -and $null -eq $r.median) 'MeasureLlama records a load failure at 65k as the figure'
  Canned '0xffff0000' '0xffff0000' 'p seed 1: the request failed (TimeoutError)'
  $r = MeasureLlama (Join-Path $tmp3 'd10') $plan 65536 $true
  Expect ($r.status -eq 'fail') 'MeasureLlama fails a request error at 65k'
  Canned '' '' 'no llama.cpp device named X; it lists: Vulkan0: Y'
  $r = MeasureLlama (Join-Path $tmp3 'd11') $plan 65536 $true
  Expect ($r.status -eq 'fail' -and $r.problem -like 'no llama.cpp device*') "MeasureLlama fails the script's own refusal at 65k and names it"
  # An error that names affinity is an affinity problem only when the server's mask could not be read.
  Canned '' '' "could not read the server's affinity (OSError)"
  $r = MeasureLlama (Join-Path $tmp3 'd11a') $plan 65536 $true
  Expect ($r.status -eq 'fail' -and $r.problem -eq 'affinity') "MeasureLlama reports an unreadable server mask as an affinity problem"
  Canned '' '' 'asked for affinity 0xffff0000, but it reads back as 0x3'
  $r = MeasureLlama (Join-Path $tmp3 'd11b') $plan 65536 $true
  Expect ($r.status -eq 'fail' -and $r.problem -like 'asked for affinity*' -and $r.error -like 'asked for affinity*') 'MeasureLlama reports a script error that names affinity as the error it is, not an affinity problem'
  Remove-Item -LiteralPath $script:cannedPath
  $r = MeasureLlama (Join-Path $tmp3 'd12') $plan 0 $false
  Expect ($r.status -eq 'fail' -and $r.problem -like '*wrote no result*') 'MeasureLlama fails when the script wrote no result'

  # MeasureLlamaSearch against a stub that copies canned search.json / best.json into the --out directory (%7).
  $stubSearch = Join-Path $tmp3 'search.cmd'
  Set-Content -LiteralPath $stubSearch -Encoding ascii -Value "@echo off`r`necho stub progress line`r`nif exist `"%~dp0c-search.json`" copy /y `"%~dp0c-search.json`" `"%7\search.json`" >nul`r`nif exist `"%~dp0c-best.json`" copy /y `"%~dp0c-best.json`" `"%7\best.json`" >nul`r`nexit /b %STUB_EXIT%"
  $python = $stubSearch
  $LlamaSearchMinutes = 75
  $env:STUB_EXIT = '0'
  Set-Content -LiteralPath (Join-Path $tmp3 'c-search.json') -Encoding utf8 -Value '{"variants":[{"label":"a"},{"label":"b"},{"label":"c"}],"budget_hit":true}'
  Set-Content -LiteralPath (Join-Path $tmp3 'c-best.json') -Encoding utf8 -Value '{"best":{"args":["-t","8"],"affinity":"0xffff0000","median":120.5,"runs":[]},"device":{"id":"Vulkan1","name":"X"},"finalists":[],"driver":"d","llama_build":"b"}'
  $r = MeasureLlamaSearch (Join-Path $tmp3 's1') 'd' 'b'
  Expect ($r -is [hashtable]) "MeasureLlamaSearch returns one record; the script's console lines don't join it"
  Expect ($r.status -eq 'ran' -and $r.best.median -eq 120.5 -and $r.variants -eq 3 -and $r.budget_hit -eq $true -and $r.device.id -eq 'Vulkan1' -and $r.path -like '*s1*best.json') 'MeasureLlamaSearch records best, device, budget_hit, the variant count and the best.json path'
  Remove-Item -LiteralPath (Join-Path $tmp3 'c-best.json')
  Set-Content -LiteralPath (Join-Path $tmp3 'c-search.json') -Encoding utf8 -Value '{"variants":[],"error":"no llama.cpp device named X; it lists: Vulkan0: Y"}'
  $env:STUB_EXIT = '1'
  $r = MeasureLlamaSearch (Join-Path $tmp3 's2') 'd' 'b'
  Expect ($r.status -eq 'fail' -and $r.problem -like 'no llama.cpp device*' -and -not $r.path) "MeasureLlamaSearch fails the script's own refusal and names it"
  $env:STUB_EXIT = '0'
  Set-Content -LiteralPath (Join-Path $tmp3 'c-search.json') -Encoding utf8 -Value '{"variants":[{"label":"a"}],"budget_hit":false}'
  $r = MeasureLlamaSearch (Join-Path $tmp3 's3') 'd' 'b'
  Expect ($r.status -eq 'fail' -and $r.problem -like '*no best.json*' -and $r.variants -eq 1) 'MeasureLlamaSearch fails when no best.json was written'
  Remove-Item -LiteralPath (Join-Path $tmp3 'c-search.json')
  $r = MeasureLlamaSearch (Join-Path $tmp3 's4') 'd' 'b'
  Expect ($r.status -eq 'fail' -and $r.problem -like '*wrote no result*') 'MeasureLlamaSearch fails when the script wrote nothing'
} finally {
  Remove-Item Env:STUB_EXIT -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $tmp3 -Recurse -Force -ErrorAction SilentlyContinue
}

# SamePositions and AttnDepthJudge (the attn-depth block's verdict): pure, no stubs.
Expect (SamePositions @('a', 'b') @('a', 'b')) 'SamePositions: equal lists are the same'
Expect (-not (SamePositions @('a', 'b') @('a', 'c'))) 'SamePositions: a differing fingerprint is not the same'
Expect (-not (SamePositions @('a') @('a', 'b'))) 'SamePositions: lists of different lengths are not the same'
Expect (-not (SamePositions @() @())) 'SamePositions: two empty lists (no fingerprints taken) are not the same'
Expect (-not (SamePositions $null $null)) 'SamePositions: two missing lists are not the same'
function AttnRow([int]$Depth, [double]$Base, [double]$New, [bool]$Equal = $true) {
  return @{ depth = $Depth; base_tok_s = $Base; new_tok_s = $New; fingerprints_equal = $Equal }
}
$good = @((AttnRow 4096 100 100), (AttnRow 27000 100 99), (AttnRow 65000 100 110))
$v = AttnDepthJudge $good 'A'
Expect ($v.status -eq 'pass' -and $v.problems.Count -eq 0 -and [math]::Abs($v.gain_65k - 1.1) -lt 1e-9) 'AttnDepthJudge passes equal, 0.99 and 1.10 ratios under A and reports gain_65k 1.1'
$v = AttnDepthJudge @((AttnRow 4096 100 98), (AttnRow 27000 100 98), (AttnRow 65000 100 100)) 'A'
Expect ($v.status -eq 'pass') 'AttnDepthJudge passes ratios of exactly 0.98 at 4,096 and 27,000 and exactly 1 at 65,000'
$v = AttnDepthJudge @((AttnRow 4096 100 97.9), (AttnRow 27000 100 100), (AttnRow 65000 100 100)) 'A'
Expect ($v.status -eq 'fail' -and $v.problems.Count -eq 1 -and $v.problems[0] -like 'depth 4096*') 'AttnDepthJudge fails 0.979 at 4,096'
$v = AttnDepthJudge @((AttnRow 4096 100 100), (AttnRow 27000 100 97), (AttnRow 65000 100 100)) 'B'
Expect ($v.status -eq 'fail' -and $v.problems[0] -like 'depth 27000*') 'AttnDepthJudge fails 0.97 at 27,000 under B too'
$v = AttnDepthJudge @((AttnRow 4096 100 100), (AttnRow 27000 100 100), (AttnRow 65000 100 99.9)) 'A'
Expect ($v.status -eq 'fail' -and $v.problems[0] -like 'depth 65000*') 'AttnDepthJudge fails a 65,000 ratio under 1 (the 0.98 allowance does not apply there)'
$mixed = @((AttnRow 4096 100 100 $true), (AttnRow 27000 100 100 $false), (AttnRow 65000 100 120 $true))
$v = AttnDepthJudge $mixed 'A'
Expect ($v.status -eq 'fail' -and $v.problems.Count -eq 1 -and $v.problems[0] -like 'depth 27000*fingerprints*') 'AttnDepthJudge under A fails differing fingerprints'
$v = AttnDepthJudge $mixed 'B'
Expect ($v.status -eq 'pass' -and $v.problems.Count -eq 0) 'AttnDepthJudge under B passes differing fingerprints (reported only)'
$v = AttnDepthJudge @((AttnRow 4096 100 100), (AttnRow 27000 0 100), (AttnRow 65000 100 100)) 'A'
Expect ($v.status -eq 'fail' -and $v.problems[0] -like 'depth 27000*missing or zero*') 'AttnDepthJudge fails a zero base rate'
$v = AttnDepthJudge @((AttnRow 4096 100 100), (AttnRow 27000 0 100 $false), (AttnRow 65000 100 100)) 'A'
Expect ($v.status -eq 'fail' -and $v.problems.Count -eq 2 -and @($v.problems -like '*fingerprints differ*').Count -eq 1) 'AttnDepthJudge still reports differing fingerprints on a row with a zero rate (stage A)'
$v = AttnDepthJudge @((AttnRow 4096 100 100), (AttnRow 27000 100 100)) 'A'
Expect ($v.status -eq 'fail' -and $v.problems -contains 'depth 65000 did not run' -and $null -eq $v.gain_65k) 'AttnDepthJudge fails when no 65,000 row ran'
$ar = AttnRow 4096 100 100
$ar.affinity_problem = 'moved'
$v = AttnDepthJudge @($ar, (AttnRow 27000 100 100), (AttnRow 65000 100 100)) 'B'
Expect ($v.status -eq 'fail' -and $v.problems[0] -like 'depth 4096: affinity*') 'AttnDepthJudge fails a row with an affinity problem, under B too'
$v = AttnDepthJudge @() 'A'
Expect ($v.status -eq 'fail' -and $null -eq $v.gain_65k) 'AttnDepthJudge fails when no depth ran'

# The first-run block's pure helpers and its launch, taken out of the runner the same way. The launch runs against a
# stub server (a pwsh script, run directly and behind a .cmd): it prints setup's start text, listens on a scratch port
# after a delay, answers /health with 200 and exits when its stdin closes. No artemis, no GPU, no port 8090.
foreach ($fn in 'IsNum', 'SpecRatio', 'AtLeast', 'Median', 'AttemptOf', 'FirstRunAttempt', 'TunedApplied', 'NetworkImports', 'ChoiceIsDefault', 'LlamaDeviceName', 'SearchedConfigCurrent', 'FirstRunPrecheck', 'FirstRunJudge', 'HealthStatus', 'FirstRunLaunch', 'HandOnSpec', 'HandOnLlamaSearch') {
  $def = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $fn }, $true)
  if (-not $def) { Expect $false "run-gpu-tests.ps1 defines $fn"; continue }
  Invoke-Expression $def.Extent.Text
}
$script:SetupLimitS = 600
Import-Module (Join-Path $PSScriptRoot 'ArtemisServer.psm1') -Force

# The attempt rule: 1 plus this commit's `first-run` aggregates (not its launch, run or precheck records, not another
# commit's); the block refuses an attempt above 2.
$tmp4 = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-first-run-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp4 | Out-Null
try {
  $ev = Join-Path $tmp4 'results.jsonl'
  Expect ((FirstRunAttempt $ev 'c1') -eq 1) 'FirstRunAttempt: no evidence file is attempt 1'
  $lines = @(
    '{"test":"first-run launch","commit":"c1","attempt":1}', '{"test":"first-run run","commit":"c1","attempt":1}',
    '{"test":"first-run precheck","commit":"c1"}', '{"test":"serve-speed-spec","commit":"c1","attempt":1}',
    '{"test":"first-run","commit":"c2","attempt":1}', 'not json "first-run"')
  Set-Content -LiteralPath $ev -Value $lines -Encoding utf8
  Expect ((FirstRunAttempt $ev 'c1') -eq 1) 'FirstRunAttempt: launch, run, precheck, other-block, other-commit and unreadable lines do not count'
  Add-Content -LiteralPath $ev -Value '{"test":"first-run","commit":"c1","attempt":1,"status":"fail"}' -Encoding utf8
  Expect ((FirstRunAttempt $ev 'c1') -eq 2 -and -not ((FirstRunAttempt $ev 'c1') -gt 2)) 'FirstRunAttempt: one aggregate makes the next attempt 2, which is allowed'
  Add-Content -LiteralPath $ev -Value '{"test":"first-run","commit":"c1","attempt":2,"status":"fail"}' -Encoding utf8
  Expect ((FirstRunAttempt $ev 'c1') -eq 3 -and (FirstRunAttempt $ev 'c1') -gt 2) 'FirstRunAttempt: two aggregates make a third attempt, which the block refuses'
  Expect ((FirstRunAttempt $ev 'c2') -eq 2) 'FirstRunAttempt: another commit has its own count'
  Expect ((AttemptOf $ev 'c1' 'serve-speed-spec') -eq 2) 'AttemptOf counts the named test only'
} finally {
  Remove-Item -LiteralPath $tmp4 -Recurse -Force -ErrorAction SilentlyContinue
}

# TunedApplied: the last `tuned` line that names the block's file decides; keys applied make it applied.
$tf = 'C:\runs\first-run\tuned.toml'
function TunedLine($Path, $Applied, $Ignored) { return [pscustomobject]@{ event = 'tuned'; path = $Path; applied = $Applied; ignored = $Ignored } }
$r = TunedApplied @(TunedLine 'C:\runs\first-run\tuned.toml' @('device_uuid', 'speculation.enabled') @()) $tf
Expect ($r.tuned_applied -and $r.tuned_applied_keys.Count -eq 2 -and $r.tuned_ignored_keys.Count -eq 0) 'TunedApplied: a tuned line naming the file with applied keys is applied'
$r = TunedApplied @(TunedLine 'c:/runs/first-run/TUNED.toml' @('device_uuid') @('speculation.sources')) $tf
Expect ($r.tuned_applied -and $r.tuned_ignored_keys -contains 'speculation.sources') 'TunedApplied: the path compares in full form with case ignored, and the ignored keys are listed'
$r = TunedApplied @(TunedLine 'C:\runs\first-run\tuned.toml' @() @()) $tf
Expect (-not $r.tuned_applied) 'TunedApplied: a tuned line that applied no key (the file is missing) is not applied'
$r = TunedApplied @(TunedLine 'D:\per-user\artemis\tuned.toml' @('device_uuid') @()) $tf
Expect (-not $r.tuned_applied) "TunedApplied: a tuned line for another file (the user's own) is not this block's"
$r = TunedApplied @() $tf
Expect (-not $r.tuned_applied -and $r.tuned_lines -eq 0) 'TunedApplied: a log with no tuned line applied nothing'
$r = TunedApplied @((TunedLine 'C:\runs\first-run\tuned.toml' @() @()), (TunedLine 'C:\runs\first-run\tuned.toml' @('device_uuid') @())) $tf
Expect ($r.tuned_applied) 'TunedApplied: the last line naming the file decides'

# NetworkImports, ChoiceIsDefault, LlamaDeviceName.
Expect ((NetworkImports @('kernel32.dll', 'WINHTTP.dll', 'ws2_32.dll', 'wininet.dll')).Count -eq 2 -and (NetworkImports @('kernel32.dll', 'ws2_32.dll')).Count -eq 0 -and (NetworkImports @()).Count -eq 0) 'NetworkImports picks out WinHTTP and WinINet and nothing else (ws2_32 is the local socket)'
Expect (ChoiceIsDefault ([pscustomobject]@{ enabled = $true; sources = @('lookup', 'mtp'); priors = $null })) 'ChoiceIsDefault: speculation on, lookup and mtp, no priors'
Expect (-not (ChoiceIsDefault ([pscustomobject]@{ enabled = $true; sources = @('lookup'); priors = $null }))) 'ChoiceIsDefault: another source list is not the defaults'
Expect (-not (ChoiceIsDefault ([pscustomobject]@{ enabled = $false; sources = @('lookup', 'mtp'); priors = $null }))) 'ChoiceIsDefault: speculation off is not the defaults'
Expect (-not (ChoiceIsDefault ([pscustomobject]@{ enabled = $true; sources = @('lookup', 'mtp'); priors = [pscustomobject]@{ cost_slope_ms = 3.0 } }))) 'ChoiceIsDefault: measured priors are not the defaults'
Expect (-not (ChoiceIsDefault $null)) 'ChoiceIsDefault: no setup_done line is not the defaults'
Expect ((LlamaDeviceName ([pscustomobject]@{ id = 'Vulkan0'; name = 'X' })) -ceq 'X' -and (LlamaDeviceName 'Y') -ceq 'Y' -and $null -eq (LlamaDeviceName $null)) 'LlamaDeviceName reads a {id, name} record, a plain name and nothing'

# SearchedConfigCurrent: a searched best.json whose driver and llama_build match today's.
$tmp5 = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-searched-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp5 | Out-Null
try {
  $bj = Join-Path $tmp5 'best.json'
  Set-Content -LiteralPath $bj -Encoding utf8 -Value '{"best":{},"finalists":[],"driver":"32.0.1","llama_build":"version: 11509 (abc)"}'
  Expect (SearchedConfigCurrent $bj '32.0.1' 'version: 11509 (abc)') 'SearchedConfigCurrent: the same driver and build match'
  Expect (-not (SearchedConfigCurrent $bj '32.0.2' 'version: 11509 (abc)')) 'SearchedConfigCurrent: another driver does not match'
  Expect (-not (SearchedConfigCurrent $bj '32.0.1' 'version: 11510 (def)')) 'SearchedConfigCurrent: another llama.cpp build does not match'
  Expect (-not (SearchedConfigCurrent $bj '' 'version: 11509 (abc)')) 'SearchedConfigCurrent: a driver that could not be read never matches'
  Set-Content -LiteralPath $bj -Encoding utf8 -Value '{"best":{},"driver":"32.0.1","llama_build":"b"}'
  Expect (-not (SearchedConfigCurrent $bj '32.0.1' 'b')) 'SearchedConfigCurrent: a file with no finalists is not a searched result'
  Expect (-not (SearchedConfigCurrent (Join-Path $tmp5 'missing.json') 'd' 'b')) 'SearchedConfigCurrent: a missing file does not match'
  Set-Content -LiteralPath $bj -Encoding utf8 -Value 'not json'
  Expect (-not (SearchedConfigCurrent $bj 'd' 'b')) 'SearchedConfigCurrent: an unreadable file does not match'
} finally {
  Remove-Item -LiteralPath $tmp5 -Recurse -Force -ErrorAction SilentlyContinue
}

# HandOnSpec and HandOnLlamaSearch: only a passing record's figures are handed on.
$script:specOnFnv = $null
$specRows = @(@{ mode = 'on'; prompt = 'write'; leaf = 'w1'; fnv = 'aa' }, @{ mode = 'off'; prompt = 'write'; leaf = 'w2'; fnv = 'bb' }, @{ mode = 'on'; prompt = 'long'; leaf = 'l1'; fnv = 'cc' })
HandOnSpec @{ status = 'pass'; on_median = 137.5 } $specRows
Expect ($script:specOnMedian -eq 137.5 -and $script:specOnStatus -eq 'pass' -and $script:specOnFnv['w1'] -eq 'aa' -and $script:specOnFnv.Count -eq 1) 'HandOnSpec hands on a passing record (the on runs but not the long one)'
HandOnSpec @{ status = 'fail'; on_median = 140.0 } $specRows
Expect ($null -eq $script:specOnMedian -and $null -eq $script:specOnFnv -and $script:specOnStatus -eq 'fail') 'HandOnSpec hands on no figures from a failed record, only its status'
$script:llamaSearchBest = 'old.json'
HandOnLlamaSearch @{ status = 'ran'; path = 'new.json' }
Expect ($script:llamaSearchBest -eq 'new.json') 'HandOnLlamaSearch hands on the best.json of a record that ran'
HandOnLlamaSearch @{ status = 'fail'; path = 'new.json'; problem = 'link dropped' }
Expect ($null -eq $script:llamaSearchBest) 'HandOnLlamaSearch hands on nothing from a failed record, and clears an older path'
$script:llamaSearchBest = $null
$script:specOnMedian = $null; $script:specOnStatus = $null

# FirstRunPrecheck: a failure the inputs already decide (not an attempt).
$pl = @{ median = 120.0; device = [pscustomobject]@{ id = 'Vulkan0'; name = 'AMD Radeon RX 9070 XT' }; affinity = '0xffff'; config_source = 'search' }
function PlWith([string]$Key, $Value) { $c = $pl.Clone(); $c[$Key] = $Value; return $c }
Expect ($null -eq (FirstRunPrecheck 137.5 $pl 'AMD Radeon RX 9070 XT')) 'FirstRunPrecheck: a searched best on the named device is fine'
Expect ($null -eq (FirstRunPrecheck 137.5 (PlWith 'config_source' 'file') 'AMD Radeon RX 9070 XT')) 'FirstRunPrecheck: a searched -LlamaConfig file is fine'
Expect ((FirstRunPrecheck $null $pl 'AMD Radeon RX 9070 XT') -like 'run serve-speed-spec and speed-llama*') 'FirstRunPrecheck: no serve-speed-spec median is a precheck failure'
Expect ((FirstRunPrecheck 137.5 $null 'AMD Radeon RX 9070 XT') -like 'run serve-speed-spec and speed-llama*') 'FirstRunPrecheck: no speed-llama figure is a precheck failure'
Expect ((FirstRunPrecheck 137.5 (PlWith 'config_source' 'plan 2a') 'AMD Radeon RX 9070 XT') -like '*plan 2a*not a searched best*') 'FirstRunPrecheck: the plan 2a configuration is a precheck failure'
Expect ((FirstRunPrecheck 137.5 (PlWith 'device' ([pscustomobject]@{ id = 'Vulkan1'; name = 'Intel iGPU' })) 'AMD Radeon RX 9070 XT') -like '*Intel iGPU*-GpuName*') 'FirstRunPrecheck: another device is a precheck failure'
Expect ((FirstRunPrecheck 137.5 (PlWith 'device' $null) 'AMD Radeon RX 9070 XT') -like '*-GpuName*') 'FirstRunPrecheck: a figure with no device is a precheck failure'

# FirstRunJudge.
$leaves = 'write-1234', 'edit-1234', 'explain-1234', 'write-1235', 'edit-1235', 'explain-1235'
function FrRows([double]$Rate, [string]$Fnv = 'aa') {
  return @($leaves | ForEach-Object { @{ leaf = $_; prompt = ($_ -split '-')[0]; decode = $Rate; fnv = $Fnv; ok = $true; tuned_applied = $true } })
}
function FrSetup([double]$Seconds = 410.5, [bool]$LimitHit = $false, $Sources = @('lookup', 'mtp')) {
  return [pscustomobject]@{ seconds = $Seconds; limit_hit = $LimitHit; enabled = $true; sources = $Sources; priors = $null }
}
function FrLaunch($Setup = (FrSetup)) {
  return @{ health = $true; reason = $null; setup = $Setup; network_imports = @(); affinity_problem = $null }
}
function FrLlama([double]$Median = 120.0, $Source = 'search', $Name = 'AMD Radeon RX 9070 XT') {
  return @{ median = $Median; device = [pscustomobject]@{ id = 'Vulkan0'; name = $Name }; affinity = '0xffff'; config_source = $Source }
}
$onFnv = @{}
foreach ($l in $leaves) { $onFnv[$l] = 'aa' }
$gpu = 'AMD Radeon RX 9070 XT'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'pass' -and $j.problems.Count -eq 0 -and $j.attempt -eq 1 -and $j.test -eq 'first-run' -and $j.tuned_median -eq 135.0 -and $j.spec_on_median -eq 137.5 -and $j.llama_median -eq 120.0 -and $j.llama_config_source -ceq 'search' -and $j.llama_device -ceq $gpu -and $j.llama_affinity -ceq '0xffff') 'FirstRunJudge passes tuned 135 against on 137.5 and llama 120, and carries the names the summary reads'
Expect ([math]::Abs($j.tuned_over_hand - 135.0 / 137.5) -lt 1e-12 -and [math]::Abs($j.tuned_over_llama - 1.125) -lt 1e-12 -and $j.runs.Count -eq 6 -and $j.runs[0].mode -ceq 'tuned') 'FirstRunJudge records both ratios and the six runs (mode tuned)'
Expect ($j.choice_is_default -and @($j.identity).Count -eq 6 -and @($j.identity | Where-Object { $_.equal -eq $true }).Count -eq 6) 'FirstRunJudge reports identity when setup chose the defaults and the hashes are equal'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0 'bb') 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'pass' -and @($j.identity | Where-Object { $_.equal -eq $false }).Count -eq 6) 'FirstRunJudge: differing completion hashes are reported, never a failure'
$j = FirstRunJudge (FrLaunch (FrSetup 400.0 $false @('lookup'))) (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'pass' -and $null -eq $j.identity -and -not $j.choice_is_default) 'FirstRunJudge: no identity report when setup chose something other than the defaults'
$j = FirstRunJudge (FrLaunch) (FrRows 132.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*below 0.97*').Count -eq 1 -and -not $j.checks.hand) 'FirstRunJudge fails 0.96 of the hand-tuned median and says so'
$j = FirstRunJudge (FrLaunch) (FrRows (0.97 * 137.5)) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.checks.hand) 'FirstRunJudge: 0.97 of the hand-tuned median is enough'
$j = FirstRunJudge (FrLaunch) (FrRows (0.97 * 140.0)) 1 140.0 $onFnv (FrLlama) $gpu @() '5b'
Expect ((0.97 * 140.0) / 140.0 -lt 0.97 -and $j.status -eq 'pass' -and $j.checks.hand) 'FirstRunJudge: the bar is the product 0.97 x on, so a median on the bar passes where the rounded ratio would fall below 0.97'
$j = FirstRunJudge (FrLaunch) (FrRows (0.97 * 140.0 - 0.01)) 1 140.0 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and -not $j.checks.hand) 'FirstRunJudge: 0.01 tok/s under the product bar fails'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv (FrLlama 135.0) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*not above 1*').Count -eq 1) 'FirstRunJudge fails a tuned median that only equals llama.cpp'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv (FrLlama 120.0 'plan 2a') $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*not a searched best*').Count -eq 1) 'FirstRunJudge fails when llama.cpp ran the plan 2a configuration'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv (FrLlama 120.0 'file') $gpu @() '5b'
Expect ($j.status -eq 'pass') 'FirstRunJudge accepts a searched -LlamaConfig file as the llama.cpp configuration'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv (FrLlama 120.0 'search' 'Intel Arc iGPU') $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*not the one named by -GpuName*').Count -eq 1) 'FirstRunJudge fails llama.cpp on a device other than -GpuName'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 $null $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*no "on" median*').Count -eq 1) 'FirstRunJudge fails without the serve-speed-spec median'
$j = FirstRunJudge (FrLaunch) (FrRows 135.0) 1 137.5 $onFnv $null $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*no median to compare*').Count -eq 1) 'FirstRunJudge fails without the speed-llama median'
$j = FirstRunJudge (FrLaunch (FrSetup 600.5)) (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*over its limit*').Count -eq 1) 'FirstRunJudge fails a setup over 600 s'
$j = FirstRunJudge (FrLaunch (FrSetup 600.0)) (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'pass') 'FirstRunJudge accepts a setup of exactly 600 s'
$j = FirstRunJudge (FrLaunch (FrSetup 590.0 $true)) (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*time limit*').Count -eq 1) 'FirstRunJudge fails a setup that hit its limit'
$ln = FrLaunch; $ln.network_imports = @('winhttp.dll')
$j = FirstRunJudge $ln (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*winhttp.dll*').Count -eq 1) 'FirstRunJudge fails a network import'
$ln = FrLaunch; $ln.setup = $null
$j = FirstRunJudge $ln (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like '*no setup_done line*').Count -eq 1) 'FirstRunJudge fails a launch with no setup_done line'
$ln = @{ health = $false; reason = '/health did not answer 200 within 720 s'; setup = $null; network_imports = @(); affinity_problem = $null }
$j = FirstRunJudge $ln @() 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like 'the launch did not reach /health*720 s').Count -eq 1 -and $null -eq $j.tuned_median) 'FirstRunJudge fails a launch that never reached /health and has no runs'
$ln = FrLaunch; $ln.affinity_problem = 'moved'
$j = FirstRunJudge $ln (FrRows 135.0) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like 'the launch: moved').Count -eq 1) "FirstRunJudge fails a launch whose mask was not the runner's"
$rows = FrRows 135.0; $rows[2].ok = $false
$j = FirstRunJudge (FrLaunch) $rows 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like 'explain-1234: the run was not whole*').Count -eq 1 -and -not $j.checks.runs) 'FirstRunJudge fails a run that was not whole'
$j = FirstRunJudge (FrLaunch) $rows 1 137.5 $onFnv (FrLlama) $gpu @('explain-1234: test 7 wrote no verdict') '5b'
Expect (@($j.problems -like 'explain-1234*').Count -eq 1) 'FirstRunJudge does not repeat a run problem that already names the run'
$rows = FrRows 135.0; $rows[4].tuned_applied = $false
$j = FirstRunJudge (FrLaunch) $rows 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and @($j.problems -like 'edit-1235: its tuned line did not apply*').Count -eq 1) 'FirstRunJudge fails a run whose tuned line did not apply the file'
$j = FirstRunJudge (FrLaunch) @((FrRows 135.0)[0..4]) 1 137.5 $onFnv (FrLlama) $gpu @() '5b'
Expect ($j.status -eq 'fail' -and $null -eq $j.tuned_median -and @($j.problems -like 'explain-1235: no tuned run was made').Count -eq 1) 'FirstRunJudge fails five runs and computes no median'
$link = @{ link_dropped = $true; pcie_gen = 3; why = 'gen 3'; hygiene = @{ pcie_gen = 4 } }
$agg = WithLink (FirstRunJudge (FrLaunch) (FrRows 135.0) 2 137.5 $onFnv (FrLlama) $gpu @() '5b') $link
Expect ($agg.status -eq 'fail' -and $agg.link_dropped -eq $true -and $agg.problem -eq 'link dropped' -and $agg.attempt -eq 2) 'a dropped link fails an otherwise passing first-run aggregate through WithLink'
$ser = ($agg | ConvertTo-Json -Compress -Depth 6)
Expect ($ser -match '"test":"first-run"' -and $ser -match '"tuned_over_hand"') 'the aggregate serialises (Record uses ConvertTo-Json)'

# FirstRunLaunch against the stub server.
$tmp6 = Join-Path ([System.IO.Path]::GetTempPath()) "artemis-hygiene-launch-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp6 | Out-Null
try {
  Set-Content -LiteralPath (Join-Path $tmp6 'stub.ps1') -Encoding utf8 -Value @'
$port = [int]$env:STUB_PORT
$delay = [int]$env:STUB_DELAY
$logDir = $null
for ($i = 0; $i -lt $args.Count - 1; $i++) { if ($args[$i] -eq '--log-dir') { $logDir = $args[$i + 1] } }
New-Item -ItemType Directory -Force $logDir | Out-Null
if ($env:STUB_EXIT) { [Console]::Error.WriteLine('boom from the stub'); exit [int]$env:STUB_EXIT }
[Console]::Out.WriteLine('First run: Artemis is measuring this PC to choose its settings.'); [Console]::Out.Flush()
$buf1 = [byte[]]::new(1)
$stdin = [Console]::OpenStandardInput().ReadAsync($buf1, 0, 1)
$clock = [System.Diagnostics.Stopwatch]::StartNew()
while (-not $stdin.IsCompleted -and $clock.Elapsed.TotalSeconds -lt $delay) { Start-Sleep -Milliseconds 100 }
if (-not $stdin.IsCompleted) {
  $tcp = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
  $tcp.Start()
  while (-not $stdin.IsCompleted) {
    if ($tcp.Pending()) {
      $c = $tcp.AcceptTcpClient()
      $st = $c.GetStream()
      $buf = [byte[]]::new(4096)
      $null = $st.Read($buf, 0, 4096)
      $out = [System.Text.Encoding]::ASCII.GetBytes("HTTP/1.1 200 OK`r`nContent-Length: 0`r`nConnection: close`r`n`r`n")
      $st.Write($out, 0, $out.Length)
      $c.Close()
    } else { Start-Sleep -Milliseconds 50 }
  }
  $tcp.Stop()
}
Add-Content -LiteralPath (Join-Path $logDir 'artemis-stub.jsonl') -Value '{"event":"exit","reason":"shutdown"}'
exit 0
'@
  Set-Content -LiteralPath (Join-Path $tmp6 'stub.cmd') -Encoding ascii -Value "@echo off`r`npwsh -NoProfile -File `"%~dp0stub.ps1`" %*`r`nexit /b %ERRORLEVEL%"
  $stub = Join-Path $tmp6 'stub.cmd'
  # The stub is run as pwsh itself, so the listener's owner is the started process. The .cmd wrapper (a grandchild
  # listens) is the negative case below.
  $pw = (Get-Command pwsh).Source
  $stubPs = Join-Path $tmp6 'stub.ps1'
  function StubArgs([string]$Logs) { return @('-NoProfile', '-File', $stubPs, 'serve', '--log-dir', $Logs) }
  $runnerMask = Get-ArtemisAffinity
  $env:STUB_PORT = '18090'
  $env:STUB_DELAY = '2'
  Remove-Item Env:STUB_EXIT -ErrorAction SilentlyContinue
  $d1 = Join-Path $tmp6 'launch1'
  $l = FirstRunLaunch $pw (StubArgs (Join-Path $d1 'logs')) $d1 18090 60
  Expect ($l.health -and $l.first_launch_s -ge 1.5 -and $null -eq $l.reason) 'FirstRunLaunch: a server that answers /health after its setup delay is reached, with the time it took'
  Expect ($l.affinity -eq $runnerMask -and $l.affinity_at_stop -eq $runnerMask -and -not $l.affinity_problem) "FirstRunLaunch: the child's mask is read at the start and before the stop, equal to the runner's"
  Expect ($l.stop.graceful -and $l.stop.exit_code -eq 0) 'FirstRunLaunch: closing the child stdin stops the server gracefully'
  Expect ((Get-Content -LiteralPath $l.stdout -Raw) -like '*First run: Artemis is measuring this PC*') "FirstRunLaunch: the child's stdout is kept"
  # A mask other than the runner's.
  $runnerMask = $runnerMask -bxor ($runnerMask -band (-$runnerMask))
  if ($runnerMask -eq 0) { $runnerMask = 1 }
  $env:STUB_DELAY = '0'
  $d2 = Join-Path $tmp6 'launch2'
  $l = FirstRunLaunch $pw (StubArgs (Join-Path $d2 'logs')) $d2 18090 60
  Expect ($l.health -and $l.affinity_problem -like '*not the runner*') "FirstRunLaunch: a child that is not on the runner's mask sets affinity_problem"
  $runnerMask = Get-ArtemisAffinity
  # No /health within the limit: the server is stopped anyway (its stdin closes).
  $env:STUB_DELAY = '60'
  $d3 = Join-Path $tmp6 'launch3'
  $l = FirstRunLaunch $pw (StubArgs (Join-Path $d3 'logs')) $d3 18090 3
  Expect (-not $l.health -and $l.reason -like '*did not answer 200 within 3 s*' -and $null -ne $l.stop) 'FirstRunLaunch: a server that never answers in time is reported and stopped'
  # The 8090-style owner check: with the .cmd wrapper the listener is a grandchild, not the started process.
  $env:STUB_EXIT = ''
  Remove-Item Env:STUB_EXIT -ErrorAction SilentlyContinue
  $env:STUB_DELAY = '1'
  $d5 = Join-Path $tmp6 'launch5'
  $l = FirstRunLaunch $stub @('serve', '--log-dir', (Join-Path $d5 'logs')) $d5 18090 60
  Expect (-not $l.health -and $null -eq $l.first_launch_s -and $l.reason -like 'port 18090 is held by pid * not by the server just started (pid *') 'FirstRunLaunch: a /health answered by a process other than the started one is not a launch'
  # An early exit names its code and the stderr tail.
  $env:STUB_EXIT = '7'
  $d4 = Join-Path $tmp6 'launch4'
  $l = FirstRunLaunch $pw (StubArgs (Join-Path $d4 'logs')) $d4 18090 60
  Expect (-not $l.health -and $l.exit_code -eq 7 -and $l.reason -like '*exited with code 7*boom from the stub*') 'FirstRunLaunch: a server that exits before /health names its exit code and stderr tail'
} finally {
  Remove-Item Env:STUB_PORT, Env:STUB_DELAY, Env:STUB_EXIT -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $tmp6 -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count) {
  Write-Host "$($failures.Count) check(s) failed"
  exit 1
}
Write-Host 'all checks passed'
exit 0
