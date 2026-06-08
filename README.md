# NextEcho

> 让播客自动转录进知识库

[English README](README.en.md) | [开源合规说明](OPEN_SOURCE_COMPLIANCE.md) | [第三方许可证清单](THIRD_PARTY_LICENSES.md)

一个本地优先的音视频转写工作台：上传文件或粘贴媒体链接，输出可复现素材包。NextEcho 聚焦“让播客自动转录进知识库”，既可以给人用网页操作，也可以给 Claude Code / Codex 这类 Agent 通过 CLI 调用。

## 能力

- 本地文件与远程媒体链接转写
- 主流媒体页面解析：YouTube、B 站、小宇宙
- 纯本地链路：`curl / yt-dlp + ffmpeg + whisper-cli`
- 默认保留 `source.*`、`audio.wav`、`transcript.txt/json/srt/vtt`
- 提供轻量 HTML 工作台
- 提供轻量 Mac App 打包脚本
- 提供 Agent 可调用 CLI 与安装说明
- 提供带 Speaker 1 / Speaker 2 的访谈逐字稿能力

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

识别来源：

```bash
python -m workbench.cli resolve-sources "https://www.youtube.com/watch?v=96jN2OCOfLs" --json
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5" --json
```

转写平台页面：

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.youtube.com/watch?v=96jN2OCOfLs" --quality accurate --json
```

转写 RSS / 播客 feed：

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

访谈逐字稿：

```bash
python -m workbench.cli speaker-transcript /path/to/audio.wav --quality accurate --json
python -m workbench.cli speaker-transcript /path/to/run_xxx
```

质量档位：

- `accurate`：高精度，优先 `large-v3-turbo-q5_0`
- `fast`：更快，优先 `base`

## Token 说明

转写计算在本地完成，不调用云端 LLM 或云端 ASR，因此音视频解析本身不消耗 LLM token。

如果通过 Agent 发起任务，Agent 理解你的指令、运行命令、读取结果时会消耗少量 Agent 编排 token；但真正的转写计算仍然是本地完成。

访谈逐字稿中的 speaker diarization 也是本地推理，不消耗 LLM token；如果采用 pyannote 模型，首次下载模型需要 Hugging Face 访问令牌，但这不是 LLM/API 计费。

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

## 轻量 Mac App

构建：

```bash
bash scripts/build_mac_app.sh
```

完成后会生成：

```text
dist/NextEcho.app
```

双击后会尝试启动本地服务并打开 `http://127.0.0.1:8765`。启动日志写入 `logs/app.log`。

## 访谈逐字稿依赖

如果你只想本地直接跑、且不想配置 Hugging Face token，先装轻量 fallback：

```bash
pip install -r requirements-speakers-lite.txt
```

这会启用 `segment-clustering` 本地后端，直接输出 `Speaker 1 / Speaker 2`。

如果你想启用更稳的 `pyannote` 主方案，再额外安装：

```bash
pip install -r requirements-speakers.txt
```

并设置 Hugging Face 访问令牌：

```bash
export HF_TOKEN=your_token_here
```

PowerShell：

```powershell
$env:HF_TOKEN="your_token_here"
```

`python -m workbench.cli doctor` 会检查当前会走 `pyannote` 还是 `segment-clustering`。

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

## 开源与合规

为了让仓库在公开发布时更规范，当前仓库已经补充了以下文档：

- [OPEN_SOURCE_COMPLIANCE.md](OPEN_SOURCE_COMPLIANCE.md)：开源发布前的合规结论、风险说明、发布策略
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)：第三方依赖与许可证清单
- `NOTICE`：第三方组件与商标声明

公开前仍建议你手动完成一件事：

- 为本仓库选择并加入你自己的项目 License，例如 MIT / Apache-2.0 / GPL。这个决定会直接影响别人如何合法使用你的代码，因此不建议由自动化工具替你默认决定。
