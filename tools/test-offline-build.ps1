#Requires -Version 7.2
# Checks that a source archive (tools\make-source-archive.ps1) builds with no network and no Vulkan SDK: it extracts the
# zip to a new temp directory, gives cargo an empty CARGO_HOME, removes glslc and the SDK variables from this process,
# builds artemis.exe with --offline --frozen, runs the SPIR-V freshness test and runs artemis.exe --help. It fails on any
# error and on any build line saying a shader is stale (a stale shader means the archive needed glslc).
# Cargo reads .cargo\config.toml (or .cargo\config) from every parent of its working directory, and the temp directory
# is under the user profile, so a config there (for example a [source] or [net] section) could hide a missing vendored
# crate. The script fails with a message naming such a file before it builds anything.
param(
  [Parameter(Mandatory)][string]$Archive,
  [switch]$Keep   # keep the extracted tree and the empty cargo home
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$Archive = (Resolve-Path $Archive).Path

# The toolchain's own cargo and rustc, found while rustup and its cargo home still work. Asked from the repo so the
# version in rust-toolchain.toml applies.
Push-Location $repo
try {
  $cargo = (rustup which cargo).Trim()
  if ($LASTEXITCODE -or -not $cargo) { throw 'rustup which cargo failed' }
  $rustc = (rustup which rustc).Trim()
  if ($LASTEXITCODE -or -not $rustc) { throw 'rustup which rustc failed' }
} finally { Pop-Location }

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("artemis-offline-" + [guid]::NewGuid().ToString('N'))
$ancestor = Split-Path $work -Parent
while ($ancestor) {
  foreach ($f in '.cargo\config.toml', '.cargo\config') {
    $path = Join-Path $ancestor $f
    if (Test-Path -LiteralPath $path) {
      throw "cargo would read $path, which could change where this build finds its crates. Move it aside for the check, or run with a temp directory (TEMP) outside $ancestor"
    }
  }
  $ancestor = Split-Path $ancestor -Parent
}
$src = Join-Path $work 'src'
$cargoHome = Join-Path $work 'cargo-home'
New-Item -ItemType Directory $work, $src, $cargoHome | Out-Null

$saved = @{}
foreach ($v in 'PATH', 'CARGO_HOME', 'RUSTC', 'VULKAN_SDK', 'ARTEMIS_SKIP_SHADERS', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET_DIR',
               'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTUP_TOOLCHAIN') {
  $saved[$v] = [Environment]::GetEnvironmentVariable($v)
}
$ok = $false
try {
  Expand-Archive -Path $Archive -DestinationPath $src
  $root = Get-ChildItem $src -Directory | Select-Object -First 1
  if (-not $root -or -not (Test-Path (Join-Path $root.FullName 'Cargo.toml'))) { throw "no source tree at the top of $Archive" }
  $tree = $root.FullName
  if (-not (Test-Path (Join-Path $tree '.cargo\config.toml'))) { throw 'the archive has no .cargo\config.toml' }

  # This process only: no glslc on PATH, no SDK, an empty cargo home, the real toolchain.
  $kept = $env:PATH -split ';' | Where-Object {
    $_ -and -not (Test-Path (Join-Path $_ 'glslc.exe')) -and -not (Test-Path (Join-Path $_ 'glslc'))
  }
  $env:PATH = $kept -join ';'
  if (Get-Command glslc -ErrorAction SilentlyContinue) { throw 'glslc is still reachable on PATH' }
  foreach ($v in 'VULKAN_SDK', 'ARTEMIS_SKIP_SHADERS', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET_DIR', 'RUSTFLAGS',
                 'CARGO_ENCODED_RUSTFLAGS', 'RUSTUP_TOOLCHAIN') {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
  }
  $env:CARGO_HOME = $cargoHome
  $env:RUSTC = $rustc

  Push-Location $tree
  try {
    "building in $tree (cargo $cargo, empty CARGO_HOME $cargoHome)"
    $stale = @()
    & $cargo build --release --offline --frozen -p artemis-tools 2>&1 | ForEach-Object {
      $line = "$_"
      $line
      if ($line -match 'stale') { $stale += $line }
    }
    if ($LASTEXITCODE) { throw "cargo build failed ($LASTEXITCODE)" }
    if ($stale) { throw "the build reported a stale shader, so the archive needed glslc:`n$($stale -join "`n")" }

    & $cargo test --offline --frozen -p kernels --test spirv_fresh 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE) { throw "spirv_fresh failed ($LASTEXITCODE)" }

    $exe = Join-Path $tree 'target\release\artemis.exe'
    if (-not (Test-Path $exe)) { throw "the build left no $exe" }
    & $exe --help
    if ($LASTEXITCODE) { throw "artemis.exe --help failed ($LASTEXITCODE)" }
  } finally { Pop-Location }
  $ok = $true
  'offline build: OK'
} finally {
  # A variable that was unset goes back to unset, not to an empty string (cargo rejects an empty CARGO_TARGET_DIR).
  foreach ($v in $saved.Keys) {
    if ($null -eq $saved[$v]) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    else { Set-Item "Env:$v" -Value $saved[$v] }
  }
  if ($Keep) { "kept: $work" } else { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
}
if (-not $ok) { exit 1 }
