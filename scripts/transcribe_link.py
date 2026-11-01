#!/usr/bin/env python3
"""One-link entrypoint for producing a self-contained NextEcho result folder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def legacy_main(argv: list[str] | None = None) -> int:
    from workbench.config import load_config
    parser = argparse.ArgumentParser(description="Transcribe one media link into a complete NextEcho folder")
    parser.add_argument("url", help="Episode/page/media URL")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to an external TOML override")
    parser.add_argument("--profile", choices=["off", "balanced", "quality", "fast"], default=argparse.SUPPRESS)
    parser.add_argument("--quality", choices=["accurate", "fast"], default="accurate")
    parser.add_argument("--language", default=argparse.SUPPRESS, help="ASR language hint, for example zh, en, or auto")
    parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp", "fun-asr-nano", "sensevoice"], default=argparse.SUPPRESS)
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--output-dir", type=Path, help="Optional explicit run directory")
    parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds")
    parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default=argparse.SUPPRESS)
    parser.add_argument("--speaker-backend", default=argparse.SUPPRESS)
    parser.add_argument("--min-speakers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-speakers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--ai-style", default="")
    parser.add_argument("--json", action="store_true", help="Print the underlying JSON result")
    args = parser.parse_args(argv)
    config = load_config(args.config_path)
    # Absence is represented by argparse, so =, abbreviated and repeated
    # options keep their parsed value. Load only the selected config once.
    defaults = {"profile": ("postprocess", "profile", "fast"), "language": ("asr", "language", "zh"),
                "asr_engine": ("asr", "engine", "auto"), "speaker_mode": ("diarization", "default_mode", "auto"),
                "speaker_backend": ("diarization", "backend", "auto"), "min_speakers": ("diarization", "min_speakers", 2),
                "max_speakers": ("diarization", "max_speakers", 6)}
    for name, (section, key, fallback) in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, config.get_path(section, key, default=fallback))

    python = str(PYTHON if PYTHON.exists() else Path(sys.executable))
    command = [
        python,
        "-m",
        "workbench.cli",
        "transcribe-page",
        args.url,
        "--quality",
        args.quality,
        "--language",
        args.language,
        "--asr-engine",
        args.asr_engine,
        "--postprocess-profile",
        args.profile,
        "--max-seconds",
        str(args.max_seconds),
        "--speaker-mode",
        args.speaker_mode,
        "--speaker-backend",
        args.speaker_backend,
        "--min-speakers",
        str(args.min_speakers),
        "--max-speakers",
        str(args.max_speakers),
        "--json",
    ]
    if args.asr_model:
        command.extend(["--asr-model", args.asr_model])
    if args.config_path:
        command.extend(["--config", str(Path(args.config_path).expanduser().resolve())])
    if args.output_dir:
        command.extend(["--output-dir", str(args.output_dir.expanduser().resolve())])
    if args.glossary:
        command.extend(["--glossary", str(args.glossary.expanduser().resolve())])
    if args.ai_style:
        command.extend(["--ai-style", args.ai_style])

    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        return completed.returncode
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or not isinstance(payload.get("output_dir"), str) or not payload["output_dir"].strip():
            raise ValueError("Missing output_dir")
        output_dir = Path(payload["output_dir"]).resolve()
        if not output_dir.is_dir():
            raise ValueError("Missing output directory")
    except (ValueError, OSError):
        print("PROTOCOL_ERROR: 旧入口未返回合法JSON，请检查本地日志", file=sys.stderr)
        return 3

    request_path = output_dir / "skill_request.json"
    request_path.write_text(
        json.dumps(
            {
                "input_url": args.url,
                "profile": args.profile,
                "quality": args.quality,
                "language": args.language,
                "asr_engine": args.asr_engine,
                "asr_model": args.asr_model,
                "speaker_mode": args.speaker_mode,
                "speaker_backend": args.speaker_backend,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "config_source": config.source,
                "config_sources": list(config.source_paths),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps({**payload, "skill_request_path": str(request_path)}, ensure_ascii=False, indent=2))
    else:
        print(output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    options = arguments[:arguments.index("--")] if "--" in arguments else arguments
    if "--legacy" in options:
        arguments.remove("--legacy")
        return legacy_main(arguments)
    # One stable pipeline and receipt implementation; no second business flow.
    from workbench.cli import main as runtime_main
    if "--json" in options:
        arguments.remove("--json")
        index = arguments.index("--") if "--" in arguments else len(arguments)
        arguments[index:index] = ["--output", "json"]
    return runtime_main(["run", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
