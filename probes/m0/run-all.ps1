# Runs every measurement probe on this PC and appends to the results file (default: target\m0-results.jsonl).
# Before running: no game, Firefox closed, nothing else heavy on the GPU.
param([string]$Out = (Join-Path $PSScriptRoot '..\..\target\m0-results.jsonl'))
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$m0 = Join-Path $PSScriptRoot '..\..\target\release\m0.exe'
function Run { & $m0 --out $Out @args; if ($LASTEXITCODE -ne 0) { Write-Warning "failed: m0 $args" } }

Run limits
Run host-alloc
foreach ($mem in 'vram', 'host', 'host-cached') {
    foreach ($wgs in 16, 32, 64, 128, 256, 512, 1024) { Run read-bw --mem $mem --kib 4352 --wgs $wgs --chain 40 }
    Run read-bw --mem $mem --kib 262144 --wgs 1024 --chain 1 --reps 10
}
Run copy-bw --queue transfer
Run copy-bw --queue compute
foreach ($q in 'q4k', 'q8_0') {
    foreach ($mem in 'vram', 'host') {
        Run matvec --quant $q --mem $mem --rows 8192
        Run matvec --quant $q --mem $mem --rows 262144 --reps 100
    }
}
Run roundtrip --mode submit --n 100000
Run roundtrip --mode semaphore --n 100000
Run roundtrip --mode spin --flag-mem host --n 1000000
Run roundtrip --mode spin --flag-mem amd-uncached --n 1000000
