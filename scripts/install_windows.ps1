$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AppUrl = "http://127.0.0.1:8765"
Set-Location $RootDir

function Show-Info($Message) {
  Add-Type -AssemblyName PresentationFramework | Out-Null
  [System.Windows.MessageBox]::Show($Message, "NextEcho") | Out-Null
}

function Get-PythonCommand {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    return @("py", "-3")
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    return @("python")
  }
  return $null
}

function Install-WithWinget($PackageIds, $DisplayName) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    return $false
  }

  foreach ($packageId in $PackageIds) {
    try {
      Write-Host "Trying to install $DisplayName with winget package $packageId"
      & winget install --exact --id $packageId --accept-source-agreements --accept-package-agreements
      if ($LASTEXITCODE -eq 0) {
        return $true
      }
    } catch {
      Write-Host "winget install failed for $packageId"
    }
  }

  return $false
}

$pythonCommand = Get-PythonCommand
if ($null -eq $pythonCommand) {
  $pythonInstalled = Install-WithWinget @("Python.Python.3.11", "Python.Python.3.12") "Python 3"
  if (-not $pythonInstalled) {
    Show-Info "未检测到 Python，且自动安装失败。请先安装 Python 3.10 或更高版本，然后重新双击 Install NextEcho.bat。"
    exit 1
  }
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
  $pythonCommand = Get-PythonCommand
}

if ($null -eq $pythonCommand) {
  Show-Info "Python 已尝试安装，但当前会话还无法识别。请关闭后重新双击 Install NextEcho.bat。"
  exit 1
}

if (-not (Test-Path ".venv")) {
  if ($pythonCommand.Length -eq 1) {
    & $pythonCommand[0] -m venv .venv
  } else {
    & $pythonCommand[0] $pythonCommand[1] -m venv .venv
  }
}

$PythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  Show-Info "虚拟环境创建失败。请重新双击 Install NextEcho.bat。"
  exit 1
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  $ffmpegInstalled = Install-WithWinget @("Gyan.FFmpeg") "FFmpeg"
  if (-not $ffmpegInstalled) {
    Show-Info "NextEcho 需要 FFmpeg，但自动安装失败。请安装 FFmpeg 后重新双击 Install NextEcho.bat。"
    exit 1
  }
}

if (-not (Get-Command whisper-cli -ErrorAction SilentlyContinue)) {
  $whisperInstalled = Install-WithWinget @("ggml.whisper.cpp", "WhisperCPP.WhisperCPP") "whisper.cpp"
  if (-not $whisperInstalled) {
    Show-Info "NextEcho 需要 whisper.cpp（whisper-cli），但自动安装没有成功。请安装 whisper.cpp 并确保 whisper-cli 在 PATH 中，然后重新双击 Install NextEcho.bat。"
    exit 1
  }
}

if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
  & $PythonExe -m pip install yt-dlp
}

New-Item -ItemType Directory -Force -Path "models\whisper.cpp" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\transcriptions" | Out-Null
& $PythonExe -m workbench.cli doctor
& $PythonExe -m workbench.cli download-model base

try {
  Invoke-WebRequest -Uri $AppUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
  Start-Process -FilePath $PythonExe -ArgumentList "app.py" -WorkingDirectory $RootDir
  Start-Sleep -Seconds 3
}

Start-Process $AppUrl
Show-Info "NextEcho 安装完成，浏览器已打开。以后直接双击 Open NextEcho.bat 就可以进入本地网站。"
