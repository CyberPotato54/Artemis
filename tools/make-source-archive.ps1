#Requires -Version 7.2
# Builds a source archive of the committed tree, target\dist\artemis-src-<short sha>.zip: `git archive HEAD` plus the
# crates `cargo vendor` collects, and a .cargo\config.toml (inside the archive only) that points cargo at them. Someone
# with Rust and the MSVC linker can unzip it and build with no network and no Vulkan SDK; see docs\install.md.
# Refuses unless the working tree is clean, so the archive holds exactly what HEAD holds, and unless the committed
# SPIR-V is glslc's output (`update-spirv --check`, which needs the Vulkan SDK on the machine that makes the archive).
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
  # Untracked files count (those not ignored), so a file the archive would miss still stops it.
  $dirty = git status --porcelain --untracked-files=normal
  if ($LASTEXITCODE) { throw 'git status failed' }
  if ($dirty) { throw "the working tree is not clean; commit or remove these first:`n$($dirty -join "`n")" }
  $sha = (git rev-parse --short HEAD).Trim()
  if ($LASTEXITCODE) { throw 'git rev-parse failed' }

  # The archive carries the committed SPIR-V; recipients build from it with no SDK, so it must be glslc's output.
  # Needs the Vulkan SDK on this machine (VULKAN_SDK, default C:\VulkanSDK\1.4.363.0).
  . (Join-Path $PSScriptRoot 'ps\ReleaseFlags.ps1')
  Assert-ArtemisSpirvFresh $repo

  $dist = Join-Path $repo 'target\dist'
  $name = "artemis-src-$sha"
  $staging = Join-Path $dist $name
  $zip = Join-Path $dist "$name.zip"
  $tracked = Join-Path $dist "$name.tracked.zip"
  New-Item -ItemType Directory -Force $dist | Out-Null
  foreach ($p in $staging, $zip, $tracked) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

  try {
    # The tracked files of HEAD.
    New-Item -ItemType Directory $staging | Out-Null
    git archive --format=zip --output=$tracked HEAD
    if ($LASTEXITCODE) { throw 'git archive failed' }
    Expand-Archive -Path $tracked -DestinationPath $staging
    Remove-Item -Force $tracked

    # The crates, then the config that makes cargo read them: the source sections `cargo vendor` prints on stdout.
    $config = $null
    Push-Location $staging
    try {
      $config = cargo vendor --locked --versioned-dirs vendor
      if ($LASTEXITCODE) { throw "cargo vendor failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
    $config = ($config -join "`n") + "`n"
    if ($config -notmatch '(?m)^\[source\.crates-io\]' -or $config -notmatch '(?m)^\[source\.vendored-sources\]') {
      throw "cargo vendor printed no source config:`n$config"
    }
    $cargoDir = Join-Path $staging '.cargo'
    New-Item -ItemType Directory -Force $cargoDir | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $cargoDir 'config.toml'), $config, (New-Object System.Text.UTF8Encoding $false))

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($staging, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
  } catch {
    # A failed run leaves neither a half-made zip nor the staging tree.
    if (Test-Path $zip) { Remove-Item -Force $zip -ErrorAction SilentlyContinue }
    throw
  } finally {
    foreach ($p in $staging, $tracked) { if (Test-Path $p) { Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue } }
  }

  $item = Get-Item $zip
  "archive: $($item.FullName)"
  "size:    $($item.Length) bytes ($([math]::Round($item.Length / 1MB, 1)) MiB)"
  "sha256:  $((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower())"
} finally { Pop-Location }
