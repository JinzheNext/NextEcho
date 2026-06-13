# Agent Install Guide: NextEcho

This guide is written for an AI coding agent such as Claude Code, Codex, Cursor Agent, or another terminal-capable assistant. Follow it to install and use NextEcho, a local, no-cloud audio and video transcription workbench for the user.

## What this capability does

- Runs audio/video transcription locally with `ffmpeg + whisper.cpp`.
- Provides a local web UI at `http://127.0.0.1:8765`.
- Provides a CLI that agents can call directly.
- Resolves YouTube, Bilibili, and Xiaoyuzhou page URLs into local transcription jobs.
- Provides a speaker-attributed interview transcript mode (`Speaker 1 / Speaker 2`) for long-form conversations.
- Writes reproducible artifact folders containing source media, normalized audio, TXT, JSON, SRT, VTT, and metadata.
- Does **not** spend LLM tokens for the transcription computation. Agent orchestration may still spend a small amount of LLM context/tokens.

## First-time installation

For human users, prefer the double-click launchers in the repository root:

- macOS: `Install NextEcho.command`, then `Open NextEcho.command`
- Windows: `Install NextEcho.bat`, then `Open NextEcho.bat`

Agents should still understand and verify the underlying scripts below.

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
- Doctor prints a recommended model.

`yt-dlp` is optional but recommended. Without it, local files and direct media links still work; webpage links may fail.

## Start the web UI

For human-facing support, prefer telling the user to double-click the launcher instead of typing commands:

- macOS: `Open NextEcho.command`
- Windows: `Open NextEcho.bat`

CLI fallback:

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
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
```

When the user wants to inspect how a page URL will resolve before transcription:

```bash
python -m workbench.cli resolve-sources "https://www.youtube.com/watch?v=96jN2OCOfLs" --json
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5" --json
```

When the user gives a single platform page URL and wants the full local transcription workflow:

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.youtube.com/watch?v=96jN2OCOfLs" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.youtube.com/watch?v=96jN2OCOfLs" --model small --json
```

When the user gives a podcast RSS feed and wants the most recent episodes:

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

Quality choices:

- `accurate`: best default when the machine can handle it; uses `large-v3-turbo-q5_0`.
- `fast`: lighter fallback; uses `base`.

Model choices:

- `tiny`: lowest resource use, best for weak or older machines
- `base`: safe lightweight default for many 8GB-class machines
- `small`: more balanced quality and speed for 8GB to 16GB machines
- `medium`: better quality for 16GB+ machines
- `large-v3-turbo-q5_0`: best default quality when the machine is strong enough

Default rule:

1. Run `python -m workbench.cli doctor` if setup status is unknown.
2. Run `python -m workbench.cli list-models` to inspect the machine-specific model recommendation.
3. If the user wants explicit control, run `python -m workbench.cli download-model <model>`.
4. Use `accurate` when doctor recommends it or when the user asks for best quality.
5. Use `fast` when doctor recommends it, memory is constrained, or the user asks for speed.

Useful model commands:

```bash
python -m workbench.cli list-models
python -m workbench.cli list-models --json
python -m workbench.cli download-model base
python -m workbench.cli download-model small
```

## Agent CLI usage for interview transcripts

When the user wants a speaker-attributed transcript for a podcast or conversation, use:

```bash
python -m workbench.cli speaker-transcript /path/to/audio.wav --quality accurate --json
python -m workbench.cli speaker-transcript /path/to/run_xxx
python -m workbench.cli speaker-transcript /path/to/audio.wav --model base --json
```

The first command accepts a raw audio file and will create a workbench run if needed. The second reuses an existing single-source run directory.

This mode produces:

- `transcript.speakers.json`
- `transcript.speakers.txt`
- `transcript.speakers.md`
- `speaker_map.json`

## Extra dependencies for speaker diarization

For a lightweight local fallback that does not need a Hugging Face token:

```bash
pip install -r requirements-speakers-lite.txt
```

This enables a local `segment-clustering` backend and is enough for `Speaker 1 / Speaker 2` interview transcripts.

If you want the heavier `pyannote` route, also install:

```bash
pip install -r requirements-speakers.txt
```

Set a Hugging Face token for pyannote model download:

```bash
export HF_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

`python -m workbench.cli doctor` will report whether the speaker transcript capability is ready, and which backend will be used.

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

Recommended agent behavior for model setup:

1. Run `doctor`
2. Run `list-models`
3. Pick a model based on the user's machine and tolerance for speed vs. quality
4. Optionally pre-download it with `download-model`
5. Pass `--model <name>` when you want deterministic behavior across runs

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
- `resolve-sources` returns a platform error: keep the stderr summary; it often tells you whether the issue is network, platform throttling, or an outdated `yt-dlp`.
- First run is slow: likely model download or first model load. Run `doctor` and check model paths.
- 8GB RAM machines: start with `accurate` only if doctor says it is reasonable; otherwise use `fast`.
- Speaker transcript fails immediately: check `pyannote.audio` installation and whether `HF_TOKEN` is configured.

## Agent behavior rules

- Do not upload user media to cloud transcription services unless the user explicitly asks.
- Do not call LLM APIs for raw transcription. Use local CLI/web capability first.
- Tell the user that transcription computation is local and does not spend LLM tokens; only your natural-language orchestration may spend agent tokens.
- If the user wants a hands-on interface, start the web UI instead of forcing CLI-only usage.
