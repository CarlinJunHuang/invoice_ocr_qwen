param(
  [Parameter(Mandatory = $true)]
  [string[]]$InputImages,

  [string[]]$Modes = @("qwen3_vl_bf16", "qwen3_vl_4bit", "qwen3_5_bf16", "qwen3_5_4bit"),

  [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot "runtime\.venv311\Scripts\python.exe"

if (-not $ConfigPath) {
  $ConfigPath = Join-Path $projectRoot "configs\default.yaml"
}

. (Join-Path $PSScriptRoot "set-runtime-env.ps1")

if (-not (Test-Path $venvPython)) {
  throw "Runtime venv not found at $venvPython. Run .\scripts\bootstrap.ps1 first."
}

& $venvPython -m invoice_ocr_qwen.cli benchmark --config $ConfigPath --modes $Modes --input $InputImages
