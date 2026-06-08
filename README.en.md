# NextEcho

> Automatically turn podcasts into knowledge base entries

[中文 README](README.md) | [Open Source Compliance](OPEN_SOURCE_COMPLIANCE.md) | [Third-Party Licenses](THIRD_PARTY_LICENSES.md)

A local-first audio and video transcription workbench. NextEcho is built to automatically turn podcasts into knowledge base entries. Upload a file or paste a media URL, and it produces a reproducible artifact bundle. It is designed for both human users through a lightweight web UI and terminal agents through a CLI.

## Features

- Transcribe local files and remote media URLs
- Resolve mainstream media pages such as YouTube, Bilibili, and Xiaoyuzhou
- Fully local pipeline: `curl / yt-dlp + ffmpeg + whisper-cli`
- Preserve `source.*`, `audio.wav`, `transcript.txt/json/srt/vtt` by default
- Lightweight HTML workbench
- Lightweight macOS app wrapper script
- Agent-friendly CLI and install guide
- Speaker-attributed interview transcripts with `Speaker 1 / Speaker 2`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m workbench.cli doctor
python -m workbench.cli serve
```

Open `http://127.0.0.1:8765`

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m workbench.cli doctor
python -m workbench.cli serve
```

## Install for Agents

If you want Claude Code, Codex, Cursor Agent, or another terminal agent to install and use this tool, point it to:

```text
AGENT_INSTALL.md
```

Suggested prompt:

> Please follow AGENT_INSTALL.md to install the local transcription workbench and verify that both the web UI and CLI work.

## Cross-Platform Install Scripts

macOS:

```bash
bash scripts/install_mac.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

## CLI

Doctor:

```bash
python -m workbench.cli doctor
python -m workbench.cli doctor --json
```

Start the web UI:

```bash
python -m workbench.cli serve
```

Transcribe local files or direct media:

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality fast
```

Resolve sources first:

```bash
python -m workbench.cli resolve-sources "https://www.youtube.com/watch?v=96jN2OCOfLs" --json
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5" --json
```

Transcribe a platform page:

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.youtube.com/watch?v=96jN2OCOfLs" --quality accurate --json
```

Transcribe an RSS or podcast feed:

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

Speaker-attributed interview transcript:

```bash
python -m workbench.cli speaker-transcript /path/to/audio.wav --quality accurate --json
python -m workbench.cli speaker-transcript /path/to/run_xxx
```

Quality presets:

- `accurate`: higher accuracy, prefers `large-v3-turbo-q5_0`
- `fast`: faster, prefers `base`

## Token Notes

Transcription runs locally and does not call a cloud LLM or cloud ASR service, so the audio or video processing itself does not consume LLM tokens.

If an agent launches the task, the agent may still spend a small amount of orchestration context or tokens while understanding your request, running commands, and reading outputs. The actual transcription compute remains local.

Speaker diarization is also local. If you use pyannote, the first model download needs a Hugging Face token, but that is not LLM or API billing.

## Reuse Existing Models

If you already have Whisper models, you can reuse them instead of downloading again:

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/your/whisper-models
```

PowerShell:

```powershell
$env:TRANSCRIBE_MODEL_DIR="C:\path\to\your\whisper-models"
```

The program checks an explicit model path, `TRANSCRIBE_MODEL_DIR`, `WHISPER_MODEL_DIR`, and then the local project model directory before downloading anything.

## Lightweight macOS App

Build:

```bash
bash scripts/build_mac_app.sh
```

This generates:

```text
dist/NextEcho.app
```

When opened, it tries to start the local service and open `http://127.0.0.1:8765`. Logs are written to `logs/app.log`.

## Speaker Transcript Dependencies

If you want a local fallback without a Hugging Face token, install:

```bash
pip install -r requirements-speakers-lite.txt
```

This enables the `segment-clustering` backend and produces `Speaker 1 / Speaker 2`.

If you want the heavier pyannote route, also install:

```bash
pip install -r requirements-speakers.txt
```

Then set a Hugging Face access token:

```bash
export HF_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

`python -m workbench.cli doctor` will report whether `pyannote` or `segment-clustering` will be used.

## Output Layout

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

## Open Source and Compliance

This repository now includes:

- [OPEN_SOURCE_COMPLIANCE.md](OPEN_SOURCE_COMPLIANCE.md): release guidance, license risk notes, and compliance policy
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): dependency and license inventory
- `NOTICE`: third-party attribution and trademark notice

One release step still needs maintainer choice:

- Add the repository's own project license, such as MIT, Apache-2.0, or GPL. That decision defines how others may legally use your code and should be chosen intentionally by the maintainer.
