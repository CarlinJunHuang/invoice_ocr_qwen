param(
  [ValidateSet("hf", "vllm")]
  [string]$Kind = "hf"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = Join-Path $projectRoot ("outputs\" + $Kind)
if (-not (Test-Path $root)) {
  throw "No outputs found for $Kind"
}

$latest = Get-ChildItem $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) {
  throw "No runs found for $Kind"
}

Write-Host "Latest run: $($latest.FullName)"
Get-ChildItem $latest.FullName | Select-Object Name,Length,LastWriteTime
if (Test-Path (Join-Path $latest.FullName "raw_output.txt")) {
  Write-Host ""
  Write-Host "----- raw_output.txt -----"
  Get-Content (Join-Path $latest.FullName "raw_output.txt") -TotalCount 120
}

