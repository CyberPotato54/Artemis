# Dot-source this before any cargo command that builds a gate or release binary:
#   . (Join-Path $PSScriptRoot 'ps\ReleaseFlags.ps1'); Set-ArtemisReleaseFlags
#
# Why: rustc writes the source path of every dependency into the binary (panic locations). Cargo passes crates.io
# sources by absolute path under the builder's cargo home, for example <profile>\.cargo\registry\src\..., so a
# release would carry the builder's user name hundreds of times. The workspace's own sources are already relative.
# Cargo's own fix (profile `trim-paths`) is unstable on the pinned toolchain, and a checked-in .cargo/config.toml can't
# name each builder's cargo home, so the remap is set here for the process.
#
# The flag must be the same for every cargo command of a run (build and test alike): a change of RUSTFLAGS rebuilds
# every crate, and a gate that mixes the two measures two binaries. Builds outside these scripts don't set it, so a
# plain `cargo build` and a gate build differ in these path strings, and the first gate build after a plain one
# rebuilds the workspace.

function Get-ArtemisCargoHome {
  if ($env:CARGO_HOME) { return [IO.Path]::GetFullPath($env:CARGO_HOME) }
  return [IO.Path]::GetFullPath((Join-Path $HOME '.cargo'))
}

function Set-ArtemisReleaseFlags {
  if ($env:RUSTFLAGS -or ($env:CARGO_ENCODED_RUSTFLAGS -and $env:CARGO_ENCODED_RUSTFLAGS -ne (Get-ArtemisReleaseFlags))) {
    throw "RUSTFLAGS or CARGO_ENCODED_RUSTFLAGS is set; gate and release builds set their own (tools\ps\ReleaseFlags.ps1). Clear it and run again"
  }
  $env:CARGO_ENCODED_RUSTFLAGS = Get-ArtemisReleaseFlags
}

function Get-ArtemisReleaseFlags {
  # CARGO_ENCODED_RUSTFLAGS separates flags with 0x1f and keeps spaces in a path whole. /Brepro makes the MSVC linker
  # write a hash of the output in place of its link time and debug id, so two builds of one commit give the same bytes
  # and a release can be checked against the gate's binary by its SHA-256.
  "--remap-path-prefix=$(Get-ArtemisCargoHome)=cargo" + [char]0x1f + '-Clink-arg=/Brepro'
}

# Fails when the binary holds the builder's home directory or cargo home, as text in UTF-8 or UTF-16.
function Assert-ArtemisNoLocalPaths([string]$Exe) {
  # .NET resolves a relative path against the process's directory, not PowerShell's location: resolve it here.
  $Exe = (Resolve-Path -LiteralPath $Exe).ProviderPath
  $bytes = [IO.File]::ReadAllBytes($Exe)
  $text = [Text.Encoding]::Latin1.GetString($bytes)
  $wide = [Text.Encoding]::Unicode.GetString($bytes)
  $needles = @([IO.Path]::GetFullPath($HOME).TrimEnd('\'), (Get-ArtemisCargoHome).TrimEnd('\')) | Select-Object -Unique
  foreach ($n in $needles) {
    foreach ($form in @($n, $n.Replace('\', '/'))) {
      $hits = ([regex]::Matches($text, [regex]::Escape($form), 'IgnoreCase')).Count + ([regex]::Matches($wide, [regex]::Escape($form), 'IgnoreCase')).Count
      if ($hits) { throw "$Exe holds the local path '$form' $hits times; was it built without Set-ArtemisReleaseFlags?" }
    }
  }
}

# The DependentLoadFlags field of a PE32+ file's load configuration (what /DEPENDENTLOADFLAG sets), read straight from
# the file. Returns 0 when the file has no load configuration directory or the directory is too short to hold the field.
function Get-ArtemisDependentLoadFlags([string]$Exe) {
  $Exe = (Resolve-Path -LiteralPath $Exe).ProviderPath
  $b = [IO.File]::ReadAllBytes($Exe)
  if ([BitConverter]::ToUInt16($b, 0) -ne 0x5A4D) { throw "$Exe is not a PE file" }
  $pe = [BitConverter]::ToInt32($b, 0x3C)
  if ([BitConverter]::ToUInt32($b, $pe) -ne 0x00004550) { throw "$Exe has no PE signature" }
  $sections = [BitConverter]::ToUInt16($b, $pe + 6)
  $optSize = [BitConverter]::ToUInt16($b, $pe + 20)
  $opt = $pe + 24
  if ([BitConverter]::ToUInt16($b, $opt) -ne 0x20B) { throw "$Exe is not a PE32+ (x64) file" }
  # Data directory 10 is the load configuration; the data directories start 112 bytes into a PE32+ optional header.
  $rva = [BitConverter]::ToUInt32($b, $opt + 112 + 10 * 8)
  $size = [BitConverter]::ToUInt32($b, $opt + 112 + 10 * 8 + 4)
  # IMAGE_LOAD_CONFIG_DIRECTORY64.DependentLoadFlags is the UInt16 at 0x4E.
  if ($rva -eq 0 -or $size -lt 0x50) { return 0 }
  $table = $opt + $optSize
  for ($i = 0; $i -lt $sections; $i++) {
    $s = $table + 40 * $i
    $va = [BitConverter]::ToUInt32($b, $s + 12)
    $span = [Math]::Max([BitConverter]::ToUInt32($b, $s + 8), [BitConverter]::ToUInt32($b, $s + 16))
    if ($rva -ge $va -and $rva -lt $va + $span) {
      return [int][BitConverter]::ToUInt16($b, [int]($rva - $va + [BitConverter]::ToUInt32($b, $s + 20)) + 0x4E)
    }
  }
  throw "the load configuration RVA $rva is in no section of $Exe"
}

# Fails unless the binary carries /DEPENDENTLOADFLAG:0x800 (LOAD_LIBRARY_SEARCH_SYSTEM32): its load-time imports then
# come from System32 only (crates\tools\build.rs sets the flag).
function Assert-ArtemisDependentLoadFlags([string]$Exe) {
  $flags = Get-ArtemisDependentLoadFlags $Exe
  if ($flags -ne 0x800) {
    throw ("$Exe has DependentLoadFlags 0x{0:X}, expected 0x800; was it built without crates\tools\build.rs's link argument?" -f $flags)
  }
}

# Fails unless the committed SPIR-V is glslc's output for the committed shaders (`update-spirv --check`). Needs the
# Vulkan SDK: VULKAN_SDK defaults to the version the repository was developed against, as the gate runner does (for
# this call only).
function Assert-ArtemisSpirvFresh([string]$Repo) {
  # Set only around the cargo call; a variable that was unset is restored as unset.
  $had = Test-Path Env:VULKAN_SDK
  $old = $env:VULKAN_SDK
  if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0' }
  Push-Location $Repo
  try {
    cargo run --locked -p kernels --bin update-spirv -- --check
    if ($LASTEXITCODE) { throw "the committed SPIR-V is not glslc's output for these shaders (update-spirv --check exited $LASTEXITCODE)" }
  } finally {
    Pop-Location
    if ($had) { $env:VULKAN_SDK = $old } else { Remove-Item Env:VULKAN_SDK -ErrorAction SilentlyContinue }
  }
}
