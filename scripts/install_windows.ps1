$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "python is required. Install Python 3.10+ first."
}

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$missing = @()
foreach ($bin in @("ffmpeg", "whisper-cli")) {
  if (-not (Get-Command $bin -ErrorAction SilentlyContinue)) {
    $missing += $bin
  }
}

if ($missing.Count -gt 0) {
  Write-Host "Missing required system dependencies: $($missing -join ', ')"
  Write-Host "Suggested install options: winget install Gyan.FFmpeg; install whisper.cpp and add whisper-cli to PATH."
}

if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
  Write-Host "Optional dependency yt-dlp is missing. Webpage URL extraction may fail."
  Write-Host "Suggested install: python -m pip install yt-dlp"
}

New-Item -ItemType Directory -Force -Path "models\whisper.cpp" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\transcriptions" | Out-Null
python -m workbench.cli doctor

Write-Host "`nInstall complete. Start the web UI with:"
Write-Host "  .\.venv\Scripts\Activate.ps1; python -m workbench.cli serve"
Write-Host "Then open http://127.0.0.1:8765"
