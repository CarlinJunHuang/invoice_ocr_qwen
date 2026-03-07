param(
  [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $repoRoot "runtime"
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
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeRoot "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeRoot "logs") | Out-Null

if (-not (Test-Path $venvPython)) {
  if ($PythonPath -like "py *") {
    Invoke-Expression "$PythonPath -m venv `"$venvRoot`""
  } else {
    & $PythonPath -m venv $venvRoot
  }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install -e "$($projectRoot)[dev]"
try {
  & $venvPython -m pip install bitsandbytes
} catch {
  Write-Warning "bitsandbytes installation failed. 4-bit modes will be unavailable until bitsandbytes installs cleanly."
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Runtime venv: $venvRoot"
Write-Host "Project cache root: $runtimeRoot"
Write-Host ""
Write-Host "Next:"
Write-Host "  .\scripts\run_extract.ps1 -Mode qwen3_vl_bf16 -InputImages .\input\invoices\page1.png"
