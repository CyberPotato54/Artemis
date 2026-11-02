#Requires -Version 7.2
# Checks that the public snapshot carries nothing private. It exports the publish set (git archive of HEAD, minus the
# paths in $excluded below) to a temp folder and scans every file, as UTF-8 and as UTF-16LE (from byte offset 0 and
# from offset 1, so a string at an odd offset in a binary is found too) and ignoring case, for each marker in the
# markers file and for a Windows user-profile path (any drive letter, a colon, then "Users" between single or doubled
# backslashes, or between forward slashes). The markers file has one marker a line; a line starting with # is a comment;
# a line of the form
#   allow <path glob> <marker>
# permits that marker in the files whose path (relative, with forward slashes) matches the glob (PowerShell -like). An
# allowed marker also covers every shorter marker inside its own match: if the drive marker and a longer path under
# that drive are both markers, `allow f <the path>` permits the drive marker within that path in f, while the drive
# marker in any other path in f is still a hit.
# Then it runs gitleaks over the export (tools/scan-secrets.ps1 -Mode dir), whose redacted report follows the marker
# hits. Prints "<file>:<line>: <marker>" for each hit and exits 1, or prints "clean: <n> files" and exits 0. It removes
# the export unless -Keep is given (the kept folder is named on stderr).
# -Root <path> checks another repository (its HEAD) instead of the one this script sits in, so a checkout can check the
# new public repository without copying the script into it.
param(
  [string]$Markers,   # default: docs\superpowers\publish\markers.txt (that folder is not in the publish set)
  [string]$Root,      # default: the repository this script is in
  [switch]$Keep
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$repo = if ($Root) { (Resolve-Path -LiteralPath $Root).Path } else { Split-Path $PSScriptRoot -Parent }
if (-not $Markers) { $Markers = Join-Path $repo 'docs/superpowers/publish/markers.txt' }
if (-not (Test-Path -LiteralPath $Markers)) { throw "no markers file at $Markers" }

# What the publish set leaves out of the repository: the design records, the notes, the CI folder, the agent context
# and the secret scanner's fingerprints (they name this history's commits).
$excluded = @('docs/superpowers', 'docs/github/notes.md', '.gitea', 'CLAUDE.md', '.gitleaksignore')
$profilePath = '[A-Za-z]:(\\{1,2}|/)Users(\\{1,2}|/)'

$markerList = [System.Collections.Generic.List[string]]::new()
$allow = [System.Collections.Generic.List[object]]::new()
foreach ($raw in Get-Content -LiteralPath $Markers -Encoding utf8) {
  $line = $raw.Trim()
  if (-not $line -or $line.StartsWith('#')) { continue }
  if ($line -match '^allow\s+(\S+)\s+(.+)$') {
    $marker = $Matches[2].Trim()
    $allow.Add([pscustomobject]@{ Glob = $Matches[1]; Marker = $marker; Regex = [regex]::new([regex]::Escape($marker), 'IgnoreCase') })
  } else {
    $markerList.Add($line)
  }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('artemis-snapshot-' + [guid]::NewGuid().ToString('N'))
$tree = Join-Path $work 'tree'
New-Item -ItemType Directory $tree | Out-Null
$hits = [System.Collections.Generic.List[string]]::new()
$count = 0
$secretFailed = $false
$secretReport = @()
try {
  $zip = Join-Path $work 'export.zip'
  $spec = @('.') + ($excluded | ForEach-Object { ":(exclude)$_" })
  git -C $repo archive --format=zip -o $zip HEAD -- @spec
  if ($LASTEXITCODE) { throw 'git archive failed' }
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tree)
  Remove-Item -LiteralPath $zip

  $rootLen = (Resolve-Path -LiteralPath $tree).Path.TrimEnd('\', '/').Length + 1
  $patterns = @($markerList | ForEach-Object { [pscustomobject]@{ Name = $_; Regex = [regex]::new([regex]::Escape($_), 'IgnoreCase') } })
  $patterns += [pscustomobject]@{ Name = 'user-profile path'; Regex = [regex]::new($profilePath, 'IgnoreCase') }

  foreach ($file in Get-ChildItem -LiteralPath $tree -Recurse -File -Force) {
    $count++
    $rel = $file.FullName.Substring($rootLen).Replace('\', '/')
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $texts = @(
      [System.Text.Encoding]::UTF8.GetString($bytes),
      [System.Text.Encoding]::Unicode.GetString($bytes),
      $(if ($bytes.Length -gt 1) { [System.Text.Encoding]::Unicode.GetString($bytes, 1, $bytes.Length - 1) } else { '' })
    )
    # The allows that apply to this file, and where each allowed marker sits in each text: a shorter marker inside such
    # a span is covered by the longer allowed one, and the same marker elsewhere in the file still hits.
    $fileAllows = @($allow | Where-Object { $rel -like $_.Glob })
    $spans = @(foreach ($text in $texts) {
        , @(foreach ($a in $fileAllows) {
            foreach ($m in $a.Regex.Matches($text)) { [pscustomobject]@{ Start = $m.Index; End = $m.Index + $m.Length; Marker = $a.Marker } }
          })
      })
    foreach ($p in $patterns) {
      $allowed = $false
      foreach ($a in $fileAllows) {
        if ($a.Marker -ieq $p.Name) { $allowed = $true; break }
      }
      if ($allowed) { continue }
      $seen = [System.Collections.Generic.HashSet[int]]::new()
      for ($t = 0; $t -lt $texts.Count; $t++) {
        $text = $texts[$t]
        foreach ($m in $p.Regex.Matches($text)) {
          $end = $m.Index + $m.Length
          $covered = $false
          foreach ($sp in $spans[$t]) {
            if ($sp.Start -le $m.Index -and $end -le $sp.End -and ($sp.End - $sp.Start) -gt $m.Length) { $covered = $true; break }
          }
          if ($covered) { continue }
          $lineNo = 1
          $at = $text.IndexOf("`n")
          while ($at -ge 0 -and $at -lt $m.Index) { $lineNo++; $at = $text.IndexOf("`n", $at + 1) }
          if ($seen.Add($lineNo)) { $hits.Add("${rel}:${lineNo}: $($p.Name)") }
        }
      }
    }
  }

  # The same export through gitleaks. The export has no .gitleaksignore (the publish set leaves it out), so it must be
  # clean without one; scan-secrets.ps1 reads only the ignore file inside the folder it scans.
  $secretReport = @(& (Join-Path $PSScriptRoot 'scan-secrets.ps1') -Mode dir -Path $tree *>&1 | ForEach-Object { "$_" })
  if ($LASTEXITCODE) { $secretFailed = $true }
} finally {
  if ($Keep) { [Console]::Error.WriteLine("kept: $work") }
  else { Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue }
}

if ($hits.Count -or $secretFailed) {
  $hits
  if ($secretFailed) { $secretReport }
  exit 1
}
"clean: $count files"
exit 0
