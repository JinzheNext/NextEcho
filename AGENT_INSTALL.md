# Agent Install Guide: Local Transcription Workbench

This guide is written for an AI coding agent such as Claude Code, Codex, Cursor Agent, or another terminal-capable assistant. Follow it to install and use a local, no-cloud audio/video transcription workbench for the user.

## What this capability does

- Runs audio/video transcription locally with `ffmpeg + whisper.cpp`.
- Provides a local web UI at `http://127.0.0.1:8765`.
- Provides a CLI that agents can call directly.
- Writes reproducible artifact folders containing source media, normalized audio, TXT, JSON, SRT, VTT, and metadata.
- Does **not** spend LLM tokens for the transcription computation. Agent orchestration may still spend a small amount of LLM context/tokens.

## First-time installation

### macOS

```bash
cd /path/to/local-transcription-workbench
bash scripts/install_mac.sh
```

If required system tools are missing, prefer Homebrew:

```bash
brew install ffmpeg whisper-cpp yt-dlp
```

Then rerun:

```bash
python -m workbench.cli doctor
```

### Windows PowerShell

```powershell
cd C:\path\to\local-transcription-workbench
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

If required system tools are missing:

- Install Python 3.10+.
- Install ffmpeg and add it to PATH.
- Install whisper.cpp and ensure `whisper-cli` is on PATH.
- Optional for webpage URLs: `python -m pip install yt-dlp` or install a system `yt-dlp` binary.

Then rerun:

```powershell
python -m workbench.cli doctor
```

## Verification checklist

Run:

```bash
python -m workbench.cli doctor
```

The setup is usable when:

- `ffmpeg` is found.
- `whisper-cli` is found.
- Python dependencies are installed.
- Doctor prints a recommended quality.

`yt-dlp` is optional but recommended. Without it, local files and direct media links still work; webpage links may fail.

## Start the web UI

```bash
python -m workbench.cli serve
```

Open:

```text
http://127.0.0.1:8765
```

Use this when the user wants a visual interface for uploading files, pasting links, opening prior runs, or downloading artifacts.

## Agent CLI usage

When the user asks you to transcribe an audio/video file or media URL, prefer the local CLI:

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality accurate --json
```

Quality choices:

- `accurate`: best default when the machine can handle it; uses `large-v3-turbo-q5_0`.
- `fast`: lighter fallback; uses `base`.

Default rule:

1. Run `python -m workbench.cli doctor` if setup status is unknown.
2. Use `accurate` when doctor recommends it or when the user asks for best quality.
3. Use `fast` when doctor recommends it, memory is constrained, or the user asks for speed.

## Model policy

The tool searches for existing local models before downloading anything.

To reuse an existing model directory, set one of:

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/whisper-models
export WHISPER_MODEL_DIR=/path/to/whisper-models
```

PowerShell:

```powershell
$env:TRANSCRIBE_MODEL_DIR="C:\path\to\whisper-models"
```

If no model is found, first transcription may download the selected model locally. This download is not an LLM/API call.

## Output contract

Each run creates:

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

For agent workflows, read `manifest.json` first, then each item's `metadata.json` and `transcript.txt`.

## Troubleshooting

- Missing `ffmpeg`: install ffmpeg and ensure it is on PATH.
- Missing `whisper-cli`: install whisper.cpp and ensure `whisper-cli` is on PATH.
- Webpage URL fails: install `yt-dlp`, then retry. Some sites may require browser cookies or may block downloads.
- First run is slow: likely model download or first model load. Run `doctor` and check model paths.
- 8GB RAM machines: start with `accurate` only if doctor says it is reasonable; otherwise use `fast`.

## Agent behavior rules

- Do not upload user media to cloud transcription services unless the user explicitly asks.
- Do not call LLM APIs for raw transcription. Use local CLI/web capability first.
- Tell the user that transcription computation is local and does not spend LLM tokens; only your natural-language orchestration may spend agent tokens.
- If the user wants a hands-on interface, start the web UI instead of forcing CLI-only usage.
