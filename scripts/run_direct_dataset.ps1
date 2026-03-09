param(
  [Parameter(Mandatory = $true)]
  [string[]]$InputFiles,

  [Parameter(Mandatory = $true)]
  [string[]]$Models,

  [string]$Backend = "ollama",

  [string]$RunPrefix = "direct-batch",

  [string]$PromptFile = "",

  [int]$MaxTokens = 3072,

  [double]$TimeoutSec = 420
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$venvPython = Join-Path $repoRoot "runtime\.venv311\Scripts\python.exe"

. (Join-Path $PSScriptRoot "set-runtime-env.ps1")

if (-not (Test-Path $venvPython)) {
  throw "Runtime venv not found at $venvPython. Run .\scripts\bootstrap.ps1 first."
}

$argsList = @(
  "-m", "invoice_ocr_qwen.direct_bench.dataset",
  "--backend", $Backend,
  "--run-prefix", $RunPrefix,
  "--max-tokens", "$MaxTokens",
  "--timeout-sec", "$TimeoutSec"
)

foreach ($model in $Models) {
  $argsList += @("--model", $model)
}

foreach ($inputFile in $InputFiles) {
  $argsList += @("--input", $inputFile)
}

if ($PromptFile) {
  $argsList += @("--prompt-file", $PromptFile)
}

& $venvPython @argsList
