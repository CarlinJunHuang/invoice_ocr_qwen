param(
  [Parameter(Mandatory = $true)]
  [string]$InputImage,

  [string]$RunName = "",

  [int]$MaxTokens = 3000,

  [int]$MaxModelLen = 4096,

  [double]$GpuMemoryUtilization = 0.90,

  [int]$StartupTimeoutSeconds = 420
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputRoot = Join-Path $projectRoot "outputs\vllm"
$statusDir = Join-Path $outputRoot ("docker-check-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null

try {
  docker --version | Out-Null
} catch {
  $payload = @{
    status = "failed"
    reason = "docker_not_available"
    message = "Docker command is not available on this machine."
    input_image = (Resolve-Path $InputImage).Path
  } | ConvertTo-Json -Depth 5
  Set-Content -Path (Join-Path $statusDir "status.json") -Value $payload -Encoding UTF8
  throw "Docker command is not available on this machine."
}

$python = Join-Path $projectRoot ".local_runtime\.venv311\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Runtime venv not found. Run .\scripts\bootstrap_hf.ps1 first."
}

$weightsRoot = Join-Path $projectRoot ".local_runtime\weights\DotsOCR_1_5"
if (-not (Test-Path $weightsRoot)) {
  throw "Local weights directory not found at $weightsRoot. Run HF bootstrap and HF inference first."
}

$containerName = "dots-ocr-vllm-tryout"
cmd /c "docker rm -f $containerName >nul 2>nul"
docker run -d --rm --name $containerName -p 8000:8000 --gpus all --ipc=host -v "${weightsRoot}:/models/DotsOCR_1_5" vllm/vllm-openai:v0.11.0 `
  --model /models/DotsOCR_1_5 `
  --served-model-name model `
  --trust-remote-code `
  --chat-template-content-format string `
  --tensor-parallel-size 1 `
  --max-model-len $MaxModelLen `
  --gpu-memory-utilization $GpuMemoryUtilization `
  --max-num-seqs 1 | Out-Null

$ready = $false
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
  try {
    $response = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/v1/models" -TimeoutSec 10
    if ($response.data.Count -gt 0) {
      $ready = $true
      break
    }
  } catch {
    if (-not (docker ps --filter "name=$containerName" --format "{{.Names}}")) {
      break
    }
  }
  Start-Sleep -Seconds 10
}

if (-not $ready) {
  $logs = docker logs $containerName --tail 400 2>&1
  $payload = @{
    status = "failed"
    reason = "server_not_ready"
    message = "vLLM server did not become ready before timeout."
    startup_timeout_seconds = $StartupTimeoutSeconds
    input_image = (Resolve-Path $InputImage).Path
    max_model_len = $MaxModelLen
    gpu_memory_utilization = $GpuMemoryUtilization
  } | ConvertTo-Json -Depth 5
  Set-Content -Path (Join-Path $statusDir "status.json") -Value $payload -Encoding UTF8
  Set-Content -Path (Join-Path $statusDir "docker_logs.txt") -Value ($logs -join [Environment]::NewLine) -Encoding UTF8
  throw "vLLM server did not become ready before timeout."
}

$args = @("-m", "run_vllm_client", "--input-image", $InputImage)
if ($RunName) {
  $args += @("--run-name", $RunName)
}
$args += @("--max-tokens", $MaxTokens)
& $python @args
