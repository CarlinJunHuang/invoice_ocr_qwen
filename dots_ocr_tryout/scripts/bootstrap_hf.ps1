param()

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $projectRoot ".local_runtime"
$venvPath = Join-Path $runtimeRoot ".venv311"

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

if (-not (Test-Path $venvPath)) {
  py -3.11 -m venv $venvPath
}

$python = Join-Path $venvPath "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e .

Write-Host "HF environment ready at $venvPath"

