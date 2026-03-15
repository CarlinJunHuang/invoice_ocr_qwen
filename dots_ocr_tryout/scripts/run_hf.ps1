param(
  [Parameter(Mandatory = $true)]
  [string]$InputImage,

  [string]$RunName = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".local_runtime\.venv311\Scripts\python.exe"

if (-not (Test-Path $python)) {
  throw "Runtime venv not found. Run .\scripts\bootstrap_hf.ps1 first."
}

$args = @("-m", "run_hf", "--input-image", $InputImage)
if ($RunName) {
  $args += @("--run-name", $RunName)
}

& $python @args

