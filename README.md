# Local Transcription Workbench

一个本地优先的音视频转写工作台：上传文件或粘贴媒体链接，输出可复现素材包。它既可以给人用网页操作，也可以给 Claude Code / Codex 这类 Agent 通过 CLI 调用。

## 能力

- 本地文件与远程媒体链接转写
- 纯本地链路：`curl / yt-dlp + ffmpeg + whisper-cli`
- 默认保留 `source.*`、`audio.wav`、`transcript.txt/json/srt/vtt`
- 提供轻量 HTML 工作台
- 提供 Agent 可调用 CLI 与安装说明

## 快速运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m workbench.cli doctor
python -m workbench.cli serve
```

打开：`http://127.0.0.1:8765`

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m workbench.cli doctor
python -m workbench.cli serve
```

## 给 Agent 安装

把这个 repo 给 Claude Code / Codex / 其他 Agent，然后让它读取：

```text
AGENT_INSTALL.md
```

推荐对 Agent 说：

> 请按 AGENT_INSTALL.md 帮我安装本地转写工作台，并验证网页端和 CLI 都能使用。

## 跨平台安装脚本

macOS：

```bash
bash scripts/install_mac.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

## CLI

自检：

```bash
python -m workbench.cli doctor
python -m workbench.cli doctor --json
```

启动网页：

```bash
python -m workbench.cli serve
```

本地转写：

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality fast
```

质量档位：

- `accurate`：高精度，优先 `large-v3-turbo-q5_0`
- `fast`：更快，优先 `base`

## Token 说明

转写计算在本地完成，不调用云端 LLM 或云端 ASR，因此音视频解析本身不消耗 LLM token。

如果通过 Agent 发起任务，Agent 理解你的指令、运行命令、读取结果时会消耗少量 Agent 编排 token；但真正的转写计算仍然是本地完成。

## 模型复用

如果你已经有 Whisper 模型，可以直接复用，不必重新下载：

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/your/whisper-models
```

PowerShell：

```powershell
$env:TRANSCRIBE_MODEL_DIR="C:\path\to\your\whisper-models"
```

程序会优先寻找显式模型路径、`TRANSCRIBE_MODEL_DIR` / `WHISPER_MODEL_DIR`，再看项目本地模型目录；都没有时才下载。

## 目录产物

```text
run_xxx/
├── manifest.json
├── run_config.json
└── items/
    └── 001_<source_label>/
        ├── metadata.json
        ├── source.<ext>
        ├── audio.wav
        ├── transcript.txt
        ├── transcript.json
        ├── transcript.srt
        └── transcript.vtt
```
