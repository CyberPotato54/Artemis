# Dot-source this to get Get-ArtemisPython: the path of a working Python 3.12 or later, for the tools in tools\py (they use
# only the standard library, so a plain interpreter runs them).
#   . (Join-Path $PSScriptRoot 'ps\Python.ps1'); $python = Get-ArtemisPython
#
# Each candidate is run once, as `python -I -c "import sys; sys.exit(sys.version_info < (3, 12))"`, and the first that exits
# 0 is taken. A name that merely resolves is not enough: python.org's Python ships no python3.exe, so on that setup
# `python3` resolves to the Microsoft Store stub in WindowsApps, which prints "Python was not found" and exits 9009.
# $Candidates is a parameter so a test can name a failing one.

function Get-ArtemisPython {
  param([string[]]$Candidates = @('python3', 'python'))
  foreach ($name in $Candidates) {
    foreach ($cmd in @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)) {
      try {
        # Cleared first, so a candidate that fails to launch cannot pass on an earlier command's exit code.
        $global:LASTEXITCODE = $null
        & $cmd.Source -I -c 'import sys; sys.exit(sys.version_info < (3, 12))' *> $null
        if ($LASTEXITCODE -eq 0) { return $cmd.Source }
      } catch { }
    }
  }
  throw "no working Python 3.12 or later: install it and put python3 or python on PATH (tools\py needs nothing else). Tried: $($Candidates -join ', ')"
}
