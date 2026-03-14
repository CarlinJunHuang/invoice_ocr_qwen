param(
  [Parameter(Mandatory = $true)]
  [string[]]$InputImages,

  [string[]]$Modes = @("glm_ocr", "paddleocr_vl_v1", "paddleocr_vl_1_5", "firered_ocr"),

  [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".local_runtime\.venv311\Scripts\python.exe"

if (-not $ConfigPath) {
  $ConfigPath = Join-Path $projectRoot "configs\default.yaml"
}

. (Join-Path $PSScriptRoot "set-runtime-env.ps1")

if (-not (Test-Path $venvPython)) {
  throw "Runtime venv not found at $venvPython. Run .\scripts\bootstrap.ps1 first."
}

& $venvPython -m invoice_ocr_glm_paddle_eval.cli compare --config $ConfigPath --modes $Modes --input $InputImages
