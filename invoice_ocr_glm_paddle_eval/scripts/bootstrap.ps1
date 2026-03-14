param(
  [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $projectRoot ".local_runtime"
$venvRoot = Join-Path $runtimeRoot ".venv311"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

if (-not $PythonPath) {
  if (Test-Path "H:\Python3\python.exe") {
    $PythonPath = "H:\Python3\python.exe"
  } else {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
      $PythonPath = "py -3.11"
    } else {
      throw "Python 3.11 was not found. Install Python 3.11 first or pass -PythonPath explicitly."
    }
  }
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeRoot "hf-cache") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeRoot "torch-cache") | Out-Null

if (-not (Test-Path $venvPython)) {
  if ($PythonPath -like "py *") {
    Invoke-Expression "$PythonPath -m venv `"$venvRoot`""
  } else {
    & $PythonPath -m venv $venvRoot
  }
}

. (Join-Path $PSScriptRoot "set-runtime-env.ps1")

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install -e $projectRoot

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Runtime root: $runtimeRoot"
Write-Host "Python: $venvPython"
Write-Host ""
Write-Host "Next:"
Write-Host "  .\scripts\run_compare.ps1 -InputImages .\private_inputs\invoices\1a.png"
