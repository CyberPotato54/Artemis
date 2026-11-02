# Get-ArtemisPeImports: the DLL names a PE file imports. Dot-source this file and call the function; it defines
# nothing else.

# The DLLs a PE file imports (the import table and the delay-load table), read straight from the file.
function Get-ArtemisPeImports([string]$Path) {
  $b = [IO.File]::ReadAllBytes($Path)
  if ([BitConverter]::ToUInt16($b, 0) -ne 0x5A4D) { throw "$Path is not a PE file" }
  $pe = [BitConverter]::ToInt32($b, 0x3C)
  if ([BitConverter]::ToUInt32($b, $pe) -ne 0x00004550) { throw "$Path has no PE signature" }
  $sections = [BitConverter]::ToUInt16($b, $pe + 6)
  $optSize = [BitConverter]::ToUInt16($b, $pe + 20)
  $opt = $pe + 24
  if ([BitConverter]::ToUInt16($b, $opt) -ne 0x20B) { throw "$Path is not a PE32+ (x64) file" }
  $dirs = $opt + 112          # the data directories of a PE32+ optional header
  $table = $opt + $optSize    # the section table
  $rvaToOffset = {
    param([uint32]$rva)
    for ($i = 0; $i -lt $sections; $i++) {
      $s = $table + 40 * $i
      $va = [BitConverter]::ToUInt32($b, $s + 12)
      $size = [Math]::Max([BitConverter]::ToUInt32($b, $s + 8), [BitConverter]::ToUInt32($b, $s + 16))
      if ($rva -ge $va -and $rva -lt $va + $size) { return [int]($rva - $va + [BitConverter]::ToUInt32($b, $s + 20)) }
    }
    throw "RVA $rva is in no section of $Path"
  }
  $cstr = {
    param([int]$at)
    $end = $at
    while ($end -lt $b.Length -and $b[$end] -ne 0) { $end++ }
    if ($end -ge $b.Length) { throw "$Path has a string at offset $at with no terminator" }
    [Text.Encoding]::ASCII.GetString($b, $at, $end - $at)
  }
  $names = [System.Collections.Generic.List[string]]::new()
  # Directory 1 is the import table (20-byte entries, name RVA at +12); directory 13 is the delay-load table (32-byte
  # entries, name RVA at +4).
  foreach ($d in @(@{ Index = 1; Size = 20; Name = 12 }, @{ Index = 13; Size = 32; Name = 4 })) {
    $rva = [BitConverter]::ToUInt32($b, $dirs + 8 * $d.Index)
    if ($rva -eq 0) { continue }
    $at = & $rvaToOffset $rva
    while ($true) {
      $nameRva = [BitConverter]::ToUInt32($b, $at + $d.Name)
      if ($nameRva -eq 0) { break }
      $names.Add((& $cstr (& $rvaToOffset $nameRva)).ToLower())
      $at += $d.Size
    }
  }
  $names | Sort-Object -Unique
}
