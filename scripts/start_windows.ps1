$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AppUrl = "http://127.0.0.1:8765"
$HealthUrl = "$AppUrl/api/health"
$LogDir = Join-Path $RootDir "logs"
$PidFile = Join-Path $LogDir "server.pid"
Set-Location $RootDir

function Show-Info($Message) {
  Add-Type -AssemblyName PresentationFramework | Out-Null
  [System.Windows.MessageBox]::Show($Message, "NextEcho") | Out-Null
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & powershell -ExecutionPolicy Bypass -File "$RootDir\scripts\install_windows.ps1"
  exit 0
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command whisper-cli -ErrorAction SilentlyContinue)) {
  & powershell -ExecutionPolicy Bypass -File "$RootDir\scripts\install_windows.ps1"
  exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

try {
  Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
  $process = Start-Process -FilePath "$RootDir\.venv\Scripts\python.exe" -ArgumentList "$RootDir\scripts\server_runner.py" -WorkingDirectory $RootDir -PassThru
  Set-Content -Path $PidFile -Value $process.Id
  Start-Sleep -Seconds 3
}

Start-Process $AppUrl
