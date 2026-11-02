#Requires -Version 7.2
# Builds the release archive, target\dist\artemis-<version>-<short sha>-windows-x64.zip. <version> is the version in
# crates\tools\Cargo.toml. The archive holds one top-level folder, <name>\ (the archive's own name without .zip), with
# eight files in it:
#   artemis.exe, artemis.toml.example, INSTALL.md (docs\install.md), LICENSE-APACHE, LICENSE-MIT, MODEL.txt,
#   server.md (docs\server.md, which INSTALL.md links) and THIRD-PARTY-LICENSES.txt (the licence texts of the crates the
#   binary links, from cargo metadata).
# The folder keeps "Extract here" from spilling files into the folder the zip was saved in, where a planted DLL could sit.
# It also writes target\dist\SHA256SUMS.txt: one "<sha256>  <path>" line (LF) for the archive, for <name>/artemis.exe (the
# path it has once the archive is unzipped beside the archive) and, when target\dist\artemis-src-<short sha>.zip exists
# for the same commit, for the source archive. The file is not signed.
# It runs tools\build-release.ps1 first and stops on any failure, so a failed build leaves no archive. It refuses unless
# the working tree is clean, so the short sha names what the binary was built from.
# It also writes target\dist\url-scan.txt: the http(s) strings in the binary with their counts, and the DLLs it imports.
# That file is a report. Dependency panic messages carry project URLs, and the server's listener needs ws2_32.dll. Any
# WinHTTP or WinINet import would be a network client in the binary and is reported as a FINDING in that file.
# -ExpectBinarySha256 <hex> makes the shipped binary provably a given one (the gate's measured binary): after the build,
# the sha256 of target\release\artemis.exe must equal it (case-insensitive), or the script throws before staging.
param(
  [string]$ExpectBinarySha256
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'ps\PeImports.ps1')

# The model Artemis loads, from docs\install.md. The archive does not hold it.
$modelRepo = 'https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF'
$modelFile = 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
$modelSize = '22,663,387,424'
$modelSha = '0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b'

# Writes the scan report: every http:// or https:// string in the binary, as ASCII and as UTF-16, with its count, then
# the imported DLLs. Returns the imported DLL names.
function Write-ArtemisUrlScan([string]$Exe, [string]$Report) {
  $bytes = [IO.File]::ReadAllBytes($Exe)
  $texts = @(
    [Text.Encoding]::Latin1.GetString($bytes),
    [Text.Encoding]::Unicode.GetString($bytes, 0, $bytes.Length - ($bytes.Length % 2)),
    [Text.Encoding]::Unicode.GetString($bytes, 1, ($bytes.Length - 1) - (($bytes.Length - 1) % 2))
  )
  $counts = @{}
  foreach ($t in $texts) {
    foreach ($m in [regex]::Matches($t, 'https?://[A-Za-z0-9._~:/?#@!$&*+,;=%\[\]()-]+')) {
      $u = $m.Value.TrimEnd('.', ',', ';', ':', ')', ']')
      $counts[$u] = 1 + [int]$counts[$u]
    }
  }
  $dlls = @(Get-ArtemisPeImports $Exe)
  $lines = [System.Collections.Generic.List[string]]::new()
  $lines.Add("artemis.exe URL scan (report only)")
  $lines.Add('')
  $lines.Add("http:// and https:// strings in the binary (ASCII and UTF-16): $($counts.Count) distinct")
  foreach ($k in ($counts.Keys | Sort-Object)) { $lines.Add(("{0,5}  {1}" -f $counts[$k], $k)) }
  $lines.Add('')
  $lines.Add("DLLs the binary imports: $($dlls.Count)")
  foreach ($d in $dlls) { $lines.Add("  $d") }
  $net = @($dlls | Where-Object { $_ -match 'winhttp|wininet' })
  $lines.Add('')
  $lines.Add($(if ($net) { "FINDING: the binary imports $($net -join ', ')" } else { 'No WinHTTP or WinINet import.' }))
  [IO.File]::WriteAllText($Report, ($lines -join "`n") + "`n", (New-Object Text.UTF8Encoding $false))
  $dlls
}

# The licence notices of the crates the binary links. From `cargo metadata --locked` (no network: the local registry
# cache), take every package reachable from artemis-tools in the resolved graph, except the workspace's own, sorted by
# name. Each gets a header (name, version, license expression) and the full text of every top-level file of its
# directory whose name starts with LICENSE, LICENCE, COPYING, NOTICE or COPYRIGHT; onig_sys also gets the
# oniguruma\COPYING it builds (BSD-2-Clause, whose notice a binary must reproduce). Returns the text and the package count.
function Get-ArtemisThirdPartyLicenses {
  $json = cargo metadata --format-version 1 --locked --filter-platform x86_64-pc-windows-msvc
  if ($LASTEXITCODE) { throw 'cargo metadata failed' }
  $meta = ($json -join "`n") | ConvertFrom-Json -Depth 100
  $byId = @{}
  foreach ($p in $meta.packages) { $byId[$p.id] = $p }
  $nodes = @{}
  foreach ($n in $meta.resolve.nodes) { $nodes[$n.id] = $n }
  $root = @($meta.packages | Where-Object { $_.name -eq 'artemis-tools' -and $meta.workspace_members -contains $_.id })
  if ($root.Count -ne 1) { throw 'artemis-tools is not exactly one workspace package' }
  $seen = [System.Collections.Generic.HashSet[string]]::new()
  $todo = [System.Collections.Generic.Stack[string]]::new()
  $todo.Push($root[0].id)
  while ($todo.Count) {
    $id = $todo.Pop()
    if (-not $seen.Add($id)) { continue }
    foreach ($d in $nodes[$id].dependencies) { $todo.Push($d) }
  }
  $workspace = [System.Collections.Generic.HashSet[string]]::new([string[]]$meta.workspace_members)
  $pkgs = @($seen | Where-Object { -not $workspace.Contains($_) } | ForEach-Object { $byId[$_] } |
      Sort-Object @{ Expression = { $_.name } }, @{ Expression = { $_.version } })
  $rule = '=' * 78
  $out = [System.Text.StringBuilder]::new()
  [void]$out.Append("Third-party licence notices for the crates statically linked into artemis.exe.`n")
  [void]$out.Append("$($pkgs.Count) packages, from Cargo.lock via cargo metadata.`n")
  foreach ($p in $pkgs) {
    $dir = Split-Path $p.manifest_path -Parent
    $files = @(Get-ChildItem -LiteralPath $dir -File | Where-Object { $_.Name -match '^(?i)(LICENSE|LICENCE|COPYING|NOTICE|COPYRIGHT)' })
    if ($p.name -eq 'onig_sys') {
      $onig = @(Get-ChildItem -LiteralPath $dir -File -Recurse -Filter 'COPYING' |
          Where-Object { $_.Directory.Name -eq 'oniguruma' })
      if (-not $onig) { throw "onig_sys has no oniguruma\COPYING under $dir" }
      $files = @($files) + $onig
    }
    $files = @($files | Sort-Object FullName -Unique)
    $license = if ($p.license) { $p.license } else { '(no license field)' }
    [void]$out.Append("`n$rule`n$($p.name) $($p.version) ($license)`n$rule`n")
    if (-not $files) {
      [void]$out.Append("No licence file in the package; its license expression is $license.`n")
      continue
    }
    foreach ($f in $files) {
      $rel = $f.FullName.Substring($dir.Length).TrimStart('\')
      $text = [IO.File]::ReadAllText($f.FullName).Replace("`r`n", "`n").TrimEnd()
      [void]$out.Append("`n--- $rel ---`n$text`n")
    }
  }
  [pscustomobject]@{ Text = $out.ToString(); Packages = $pkgs.Count }
}

Push-Location $repo
try {
  $dirty = git status --porcelain --untracked-files=normal
  if ($LASTEXITCODE) { throw 'git status failed' }
  if ($dirty) { throw "the working tree is not clean; commit or remove these first:`n$($dirty -join "`n")" }
  $sha = (git rev-parse --short HEAD).Trim()
  if ($LASTEXITCODE) { throw 'git rev-parse failed' }
  $cargoToml = Get-Content -Raw (Join-Path $repo 'crates\tools\Cargo.toml')
  if ($cargoToml -notmatch '(?m)^version\s*=\s*"([^"]+)"') { throw 'no version in crates\tools\Cargo.toml' }
  $version = $Matches[1]

  # The build. Any failure throws here, before anything is staged or zipped.
  & (Join-Path $PSScriptRoot 'build-release.ps1')
  $exe = Join-Path $repo 'target\release\artemis.exe'
  if (-not (Test-Path $exe)) { throw "$exe is missing after the build" }
  if ($ExpectBinarySha256) {
    $built = (Get-FileHash $exe -Algorithm SHA256).Hash
    if ($built -ine $ExpectBinarySha256.Trim()) {
      throw "the built artemis.exe has sha256 $($built.ToLower()), expected $($ExpectBinarySha256.Trim().ToLower()); nothing was staged"
    }
  }

  # The notices of every crate the binary links, before anything is staged.
  $licenses = Get-ArtemisThirdPartyLicenses
  if ($licenses.Text.Length -eq 0 -or $licenses.Text -notmatch 'onig_sys') {
    throw 'THIRD-PARTY-LICENSES.txt would be empty or would not name onig_sys'
  }

  $dist = Join-Path $repo 'target\dist'
  $name = "artemis-$version-$sha-windows-x64"
  $staging = Join-Path $dist $name
  $zip = Join-Path $dist "$name.zip"
  $scan = Join-Path $dist 'url-scan.txt'
  New-Item -ItemType Directory -Force $dist | Out-Null
  foreach ($p in $staging, $zip, $scan, (Join-Path $dist 'SHA256SUMS.txt')) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

  New-Item -ItemType Directory $staging | Out-Null
  Copy-Item $exe (Join-Path $staging 'artemis.exe')
  Copy-Item (Join-Path $repo 'tools\config\artemis.toml.example') $staging
  Copy-Item (Join-Path $repo 'docs\install.md') (Join-Path $staging 'INSTALL.md')
  Copy-Item (Join-Path $repo 'docs\server.md') (Join-Path $staging 'server.md')
  Copy-Item (Join-Path $repo 'LICENSE-APACHE') $staging
  Copy-Item (Join-Path $repo 'LICENSE-MIT') $staging
  $model = @(
    'The model Artemis loads. It is not part of this archive; download it once.'
    ''
    'Model:      Qwen3.6-35B-A3B, Q4_K_M quantization (Unsloth)'
    "Repository: $modelRepo"
    "File:       $modelFile"
    "Size:       $modelSize bytes"
    "SHA-256:    $modelSha"
    ''
    'INSTALL.md describes the download and how to check the file.'
  ) -join "`r`n"
  [IO.File]::WriteAllText((Join-Path $staging 'MODEL.txt'), $model + "`r`n", (New-Object Text.UTF8Encoding $false))

  [IO.File]::WriteAllText((Join-Path $staging 'THIRD-PARTY-LICENSES.txt'), $licenses.Text, (New-Object Text.UTF8Encoding $false))

  $dlls = Write-ArtemisUrlScan $exe $scan

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  try {
    [IO.Compression.ZipFile]::CreateFromDirectory($staging, $zip, [IO.Compression.CompressionLevel]::Optimal, $true)
  } catch {
    # A failed zip must not stay behind looking like a release.
    if (Test-Path $zip) { Remove-Item -Force $zip -ErrorAction SilentlyContinue }
    throw
  }
  Remove-Item -Recurse -Force $staging

  # The archive holds exactly the eight files, all under the one folder.
  $want = @('INSTALL.md', 'LICENSE-APACHE', 'LICENSE-MIT', 'MODEL.txt', 'THIRD-PARTY-LICENSES.txt', 'artemis.exe',
    'artemis.toml.example', 'server.md') | ForEach-Object { "$name/$_" }
  $z = [IO.Compression.ZipFile]::OpenRead($zip)
  try { $have = @($z.Entries | ForEach-Object { $_.FullName } | Sort-Object -CaseSensitive) } finally { $z.Dispose() }
  if (($have -join '|') -cne (($want | Sort-Object -CaseSensitive) -join '|')) {
    Remove-Item -Force $zip
    throw "the archive holds [$($have -join ', ')], expected [$($want -join ', ')]"
  }

  # SHA256SUMS.txt beside the archive. The source archive is listed only when it was made for this commit.
  $sumsPath = Join-Path $dist 'SHA256SUMS.txt'
  # Each line names a path relative to target\dist as the files sit after the archive is unzipped there, so
  # `sha256sum -c SHA256SUMS.txt` run in that folder checks them: the archive, <name>/artemis.exe and the source archive.
  $srcZip = Join-Path $dist "artemis-src-$sha.zip"
  $sumFiles = [ordered]@{ "$name.zip" = $zip; "$name/artemis.exe" = $exe }
  if (Test-Path $srcZip) { $sumFiles["artemis-src-$sha.zip"] = $srcZip }
  $sums = @{}
  foreach ($k in $sumFiles.Keys) { $sums[$k] = (Get-FileHash $sumFiles[$k] -Algorithm SHA256).Hash.ToLower() }
  $sumNames = [string[]]@($sums.Keys)
  [Array]::Sort($sumNames, [StringComparer]::Ordinal)
  $sumText = (($sumNames | ForEach-Object { "$($sums[$_])  $_`n" }) -join '')
  [IO.File]::WriteAllText($sumsPath, $sumText, (New-Object Text.UTF8Encoding $false))

  $item = Get-Item $zip
  "archive:        $($item.FullName)"
  "size:           $($item.Length) bytes ($([math]::Round($item.Length / 1MB, 1)) MiB)"
  "archive sha256: $((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower())"
  "artemis.exe sha256: $((Get-FileHash $exe -Algorithm SHA256).Hash.ToLower())"
  "third-party:    $($licenses.Packages) packages, $($licenses.Text.Length) characters of notices"
  "url scan:       $scan ($($dlls.Count) imported DLLs)"
  "checksums:      $sumsPath ($($sums.Count) files)"
} finally { Pop-Location }
