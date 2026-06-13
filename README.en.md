# NextEcho

> Automatically turn podcasts into knowledge base entries

[中文 README](README.md) | [Open Source Compliance](OPEN_SOURCE_COMPLIANCE.md) | [Third-Party Licenses](THIRD_PARTY_LICENSES.md)

NextEcho is a local-first audio and video transcription workbench. You can upload local files or paste YouTube, Bilibili, Xiaoyuzhou, and podcast RSS links, then turn them into reusable text assets with a structured artifact bundle on disk.

It is designed for two kinds of users:

- Human users: work directly in the web UI
- Agents: integrate through the CLI or by launching the local web workflow

## What It Is

NextEcho is useful when you want to:

- turn podcasts, videos, and interviews into full transcripts
- move an episode into your knowledge base or note system
- keep source media, extracted audio, subtitles, and intermediate outputs for reuse
- give Claude Code, Codex, Cursor Agent, or similar tools a reliable local transcription capability

Core characteristics:

- local-first, with no dependency on cloud ASR
- supports both local files and remote links
- supports both web and command-line workflows
- preserves `source.*`, `audio.wav`, `transcript.txt/json/srt/vtt` by default
- transcript text includes project attribution by default so shared copies keep their source

## Supported Inputs

- Local files: `mp3`, `mp4`, `m4a`, `wav`, `flac`, `aac`, `mov`, `webm`
- Platform pages: YouTube, Bilibili, Xiaoyuzhou
- Direct media URLs: audio or video file links
- RSS and podcast feeds

## Recommended User Path

If you are a human user, the recommended path is a double-click flow with no terminal typing:

### macOS

1. Double-click `Install NextEcho.command`
2. After setup finishes, double-click `Open NextEcho.command` or open `dist/NextEcho.app`

### Windows

1. Double-click `Install NextEcho.bat`
2. After setup finishes, double-click `Open NextEcho.bat`

These one-click launchers try to handle the following automatically:

- create the Python virtual environment
- install Python dependencies
- check and try to install `ffmpeg` and `whisper-cli`
- pre-download the default `base` model
- start the local site
- open the browser at `http://127.0.0.1:8765`

## Installation

### macOS

The fastest path is the double-click installer:

- `Install NextEcho.command`

Under the hood, it calls:

```bash
bash scripts/install_mac.sh
```

That script will:

- check whether Python 3 is available
- create `.venv`
- install `requirements.txt`
- check `ffmpeg` and `whisper-cli`
- auto-install missing system tools through Homebrew when available
- pre-download the `base` model
- start the local site and open the browser

If you prefer to install manually, the minimum setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg whisper-cpp yt-dlp
python -m workbench.cli doctor
```

### Windows PowerShell

The recommended path is the double-click installer:

- `Install NextEcho.bat`

Under the hood, it calls:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

For a manual setup, make sure that:

- Python 3.10+ is installed
- `ffmpeg` is on `PATH`
- `whisper-cli` is on `PATH`
- `yt-dlp` is installed if you want platform-page resolution

Then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m workbench.cli doctor
```

### Verify the Setup First

```bash
python -m workbench.cli doctor
```

The main pipeline is ready when:

- `ffmpeg` is found
- `whisper-cli` is found
- at least one Whisper model is available, or first-run download is allowed

### How to Choose a Model

If you want to choose a model explicitly based on the machine instead of relying on the default:

1. Inspect the machine recommendation first:

```bash
python -m workbench.cli doctor
python -m workbench.cli list-models
```

2. Pick a model size:

- `tiny`: lowest resource usage, best for older machines or quick previews
- `base`: lightweight and stable, a good fit for roughly 8GB machines
- `small`: more balanced accuracy and speed, a good fit for 8GB to 16GB machines
- `medium`: higher accuracy, best for 16GB+ machines
- `large-v3-turbo-q5_0`: the default high-accuracy model for stronger machines

3. Download the exact model you want:

```bash
python -m workbench.cli download-model base
python -m workbench.cli download-model small
python -m workbench.cli download-model large-v3-turbo-q5_0
```

4. Use that model explicitly during transcription:

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

## For Human Users

### Option 1: Use the Web App

This is the best default for most people.

1. Double-click `Open NextEcho.command` on macOS or `Open NextEcho.bat` on Windows
2. The browser will open `http://127.0.0.1:8765`

3. Choose one input method in the UI:

- upload local audio or video files
- paste one or more links, one per line

4. Choose a quality mode:

- `accurate`: recommended by default, prefers `large-v3-turbo-q5_0`
- `fast`: speed-first mode, uses `base`

5. Start the transcription and wait for the outputs.

The web workflow is ideal when:

- you want to transcribe one episode quickly
- you do not want to manage CLI flags
- you want to inspect recent runs and artifact paths visually

### Option 2: Open the macOS App

If you want a more app-like startup flow on macOS:

```bash
bash scripts/build_mac_app.sh
```

This generates:

```text
dist/NextEcho.app
```

When opened, it will try to:

- check the local environment
- trigger the installer automatically if dependencies are missing
- start the service
- open `http://127.0.0.1:8765`

Logs are written to:

```text
logs/app.log
```

### Option 3: Use the CLI Directly

This is best for scripting, batch work, and knowledge-workflow automation.

Most common commands:

#### 1. Transcribe a local file

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
```

#### 2. Transcribe a direct media URL

```bash
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality fast
```

#### 3. Resolve a source before deciding whether to transcribe it

```bash
python -m workbench.cli resolve-sources "https://www.youtube.com/watch?v=96jN2OCOfLs" --json
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --json
```

#### 4. Transcribe a platform page directly

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

#### 5. Transcribe a podcast RSS feed

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

#### 6. Inspect supported models for the current machine

```bash
python -m workbench.cli list-models
python -m workbench.cli list-models --json
```

#### 7. Pre-download a specific model

```bash
python -m workbench.cli download-model base
python -m workbench.cli download-model large-v3-turbo-q5_0
```

## For Agents

NextEcho can serve agents through either the CLI or a launched local web workflow.

### Option 1: Let the Agent Use the CLI

This is the most reliable integration path. Give the repository to the agent and point it to:

```text
AGENT_INSTALL.md
```

Suggested prompt:

> Please follow AGENT_INSTALL.md to install NextEcho and verify that both the web UI and CLI work.

Common agent-side commands:

#### 1. Environment check first

```bash
python -m workbench.cli doctor
python -m workbench.cli doctor --json
python -m workbench.cli list-models --json
```

#### 2. Transcribe a local file or direct media URL

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality accurate --json
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
```

#### 3. Resolve a platform page first

```bash
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --json
```

#### 4. Transcribe a platform page directly

```bash
python -m workbench.cli transcribe-page "https://www.youtube.com/watch?v=96jN2OCOfLs" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

#### 5. Process an RSS feed

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

Recommended agent flow:

1. Run `doctor`
2. Run `list-models` and choose between `tiny`, `base`, `small`, `medium`, or `large-v3-turbo-q5_0` for the user's machine
3. If useful, run `download-model <model>` first
4. If the input is a platform link, run `resolve-sources`
5. Choose between `transcribe`, `transcribe-page`, or `transcribe-feed`
6. Read `manifest.json` first, then each item's `metadata.json` and `transcript.txt`

### Option 2: Let the Agent Launch the Web App for the User

If the user prefers a visual flow, the agent can simply start:

```bash
python -m workbench.cli serve
```

Then direct the user to:

```text
http://127.0.0.1:8765
```

This is a good fit when:

- the user wants to paste links or upload files manually
- the agent only needs to prepare the environment and open the entry point
- the user wants to inspect the final results themselves

## Speaker-Attributed Interview Transcripts

If you want `Speaker 1 / Speaker 2` style output, you need extra speaker-diarization dependencies.

### Lightweight Local Fallback

If you do not want to configure a Hugging Face token, start with:

```bash
pip install -r requirements-speakers-lite.txt
```

This enables the `segment-clustering` backend and is good for getting the workflow running.

### Stronger pyannote Path

If you want stronger speaker diarization, also install:

```bash
pip install -r requirements-speakers.txt
```

Then set:

```bash
export HF_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

### Commands

#### 1. Generate a speaker transcript from a raw audio file

```bash
python -m workbench.cli speaker-transcript /path/to/audio.wav --quality accurate --json
python -m workbench.cli speaker-transcript /path/to/audio.wav --model base --json
```

#### 2. Continue from an existing single-source run directory

```bash
python -m workbench.cli speaker-transcript /path/to/run_xxx
```

Outputs include:

- `transcript.speakers.json`
- `transcript.speakers.txt`
- `transcript.speakers.md`
- `speaker_map.json`

## Output Location

Each run creates a `run_xxx/` directory like this:

```text
outputs/transcriptions/run_xxx/
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

If you also generate a speaker transcript, you will usually see these files at the run root:

- `transcript.speakers.json`
- `transcript.speakers.txt`
- `transcript.speakers.md`
- `speaker_map.json`

## Reusing Existing Models

If you already have Whisper models, you can reuse them instead of downloading again:

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/your/whisper-models
```

PowerShell:

```powershell
$env:TRANSCRIBE_MODEL_DIR="C:\path\to\your\whisper-models"
```

The program checks, in order:

- an explicit model path
- `TRANSCRIBE_MODEL_DIR`
- `WHISPER_MODEL_DIR`
- the local project model directory

If none are available, it will download the required model on first run.

## Token and Cost Notes

- transcription computation runs locally and does not call a cloud LLM or cloud ASR service
- media parsing itself does not consume LLM tokens
- if an agent launches the task, the agent may still consume a small amount of orchestration context or tokens while interpreting the request, running commands, and reading outputs
- if you use pyannote, the first model download may require a Hugging Face token, but that is not LLM or API billing

## Open Source and Compliance

The repository includes:

- [OPEN_SOURCE_COMPLIANCE.md](OPEN_SOURCE_COMPLIANCE.md): release guidance, license risk notes, and compliance policy
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): dependency and license inventory
- `NOTICE`: third-party attribution and trademark notice

This repository is licensed under:

- `AGPL-3.0`

In practice, that means:

- others may use, modify, and redistribute the project
- if they modify it and provide it as a network service, they must make the corresponding source available under the AGPL terms
