$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runtimeRoot = Join-Path $repoRoot "runtime"

$env:HF_HOME = Join-Path $runtimeRoot "hf-cache"
$env:HF_HUB_CACHE = Join-Path $runtimeRoot "hf-cache"
$env:TRANSFORMERS_CACHE = Join-Path $runtimeRoot "hf-cache"
$env:TORCH_HOME = Join-Path $runtimeRoot "torch-cache"
$env:INVOICE_OCR_QWEN_REPO_ROOT = $repoRoot
