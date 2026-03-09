param(
  [Parameter(Mandatory = $true)]
  [string]$InputFile,

  [Parameter(Mandatory = $true)]
  [string]$Model,

  [string]$Backend = "ollama",

  [string]$RunName = "",

  [string]$PromptFile = "",

  [int]$MaxTokens = 3072,

  [double]$TimeoutSec = 420
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot "runtime\.venv311\Scripts\python.exe"

. (Join-Path $PSScriptRoot "set-runtime-env.ps1")

if (-not (Test-Path $venvPython)) {
  throw "Runtime venv not found at $venvPython. Run .\scripts\bootstrap.ps1 first."
}

$argsList = @(
  "-m", "invoice_ocr_qwen.direct_bench",
  "--backend", $Backend,
  "--model", $Model,
  "--input", $InputFile,
  "--max-tokens", "$MaxTokens",
  "--timeout-sec", "$TimeoutSec"
)

if ($RunName) {
  $argsList += @("--run-name", $RunName)
}

if ($PromptFile) {
  $argsList += @("--prompt-file", $PromptFile)
}

& $venvPython @argsList
