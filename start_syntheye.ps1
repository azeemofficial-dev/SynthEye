param(
  [int]$Port = 8000,
  [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

$pythonPath = Join-Path $scriptRoot ".venv311\\Scripts\\python.exe"
if (-not (Test-Path $pythonPath)) {
  $pythonPath = "python"
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
  $pidToStop = $listener.OwningProcess
  try {
    Stop-Process -Id $pidToStop -Force -ErrorAction Stop
    Write-Host "Stopped process on port $Port (PID=$pidToStop)"
  } catch {
    Write-Warning "Could not stop PID=$pidToStop on port $Port. You may need admin permissions."
  }
}

$args = @("-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "$Port")
if (-not $NoReload) {
  $args += "--reload"
}

Write-Host "Starting SynthEye API from: $scriptRoot"
Write-Host "Python: $pythonPath"
Write-Host "URL: http://127.0.0.1:$Port/"

& $pythonPath @args
