# Builds the oracle: llama.cpp at the pinned tag, with the CPU and Vulkan backends, plus artemis-oracle from
# dump.cpp. Everything lands under $Root, which has no default: pass the folder, and give the same one to
# tools\run-gpu-tests.ps1 as -Oracle (or set ARTEMIS_ORACLE).
param([Parameter(Mandatory)][string]$Root)
$ErrorActionPreference = 'Stop'
$Tag = 'b11509'
$Commit = '1167d3f42c2b4dc5469bf4e696cb8c678ca79110'
$CMake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0' }

$src = Join-Path $Root 'llama.cpp'
# Network only when the clone or the pinned commit is missing: a clone that holds $Commit builds offline.
if (-not (Test-Path $src)) { git clone --quiet https://github.com/ggml-org/llama.cpp $src; if ($LASTEXITCODE) { throw 'git clone failed' } }
git -C $src cat-file -e "$Commit^{commit}" 2>$null
if ($LASTEXITCODE) {
  git -C $src fetch --quiet --tags origin
  if ($LASTEXITCODE) { throw "git fetch failed, and the clone at $src lacks $Commit" }
}
git -C $src checkout --quiet $Commit
if ($LASTEXITCODE) { throw "git checkout $Commit failed" }
$head = (git -C $src rev-parse HEAD).Trim()
if ($head -ne $Commit) { throw "llama.cpp $Tag is $head, expected $Commit" }

$build = Join-Path $Root 'build'
& $CMake -S $PSScriptRoot -B $build -G 'Visual Studio 17 2022' -A x64 "-DLLAMA_DIR=$src" -DGGML_VULKAN=ON -DGGML_NATIVE=ON
if ($LASTEXITCODE) { throw 'cmake configure failed' }
& $CMake --build $build --config Release --parallel 16 --target artemis-oracle llama-tokenize llama-perplexity llama-server
if ($LASTEXITCODE) { throw 'cmake build failed' }
Write-Host "built into $build\bin\Release"
