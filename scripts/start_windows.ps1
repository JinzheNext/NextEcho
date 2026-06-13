$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AppUrl = "http://127.0.0.1:8765"
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

try {
  Invoke-WebRequest -Uri $AppUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
  Start-Process -FilePath "$RootDir\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory $RootDir
  Start-Sleep -Seconds 3
}

Start-Process $AppUrl
