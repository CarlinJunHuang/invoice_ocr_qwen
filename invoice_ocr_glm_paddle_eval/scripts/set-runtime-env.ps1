$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $projectRoot ".local_runtime"
$tmpRoot = Join-Path $runtimeRoot "tmp"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$env:HF_HOME = Join-Path $runtimeRoot "hf-cache"
$env:HF_HUB_CACHE = Join-Path $runtimeRoot "hf-cache"
$env:HF_MODULES_CACHE = Join-Path $runtimeRoot "hf-cache\\modules"
$env:TRANSFORMERS_CACHE = Join-Path $runtimeRoot "hf-cache"
$env:TORCH_HOME = Join-Path $runtimeRoot "torch-cache"
$env:XDG_CACHE_HOME = $runtimeRoot
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot
