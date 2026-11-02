# Builds the release binary, target\release\artemis.exe, with the same flags the gate runner uses, checks that it
# holds no local path and carries the System32-only load flag, and prints its sha256. It first runs
# `update-spirv --check`, which recompiles every shader and compares bytes with the committed SPIR-V, so a release build
# needs the Vulkan SDK (VULKAN_SDK, default C:\VulkanSDK\1.4.363.0) and refuses a tree whose compiled shaders are not
# glslc's output for its sources.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'ps\ReleaseFlags.ps1')
if ($env:CARGO_TARGET_DIR) { throw "CARGO_TARGET_DIR is set ('$env:CARGO_TARGET_DIR'); clear it so the binary lands in $repo\target\release" }
if ($env:ARTEMIS_SKIP_SHADERS) { throw 'ARTEMIS_SKIP_SHADERS is set; a release needs its shaders' }
Set-ArtemisReleaseFlags
Assert-ArtemisSpirvFresh $repo
Push-Location $repo
try {
  cargo build --release --locked -p artemis-tools
  if ($LASTEXITCODE) { throw "cargo build failed ($LASTEXITCODE)" }
} finally { Pop-Location }
$exe = Join-Path $repo 'target\release\artemis.exe'
Assert-ArtemisNoLocalPaths $exe
Assert-ArtemisDependentLoadFlags $exe
"artemis.exe sha256: $((Get-FileHash $exe -Algorithm SHA256).Hash.ToLower())"
