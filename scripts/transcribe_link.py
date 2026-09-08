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
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe one media link into a complete NextEcho folder")
    parser.add_argument("url", help="Episode/page/media URL")
    parser.add_argument("--profile", choices=["off", "balanced", "quality", "fast"], default="fast")
    parser.add_argument("--quality", choices=["accurate", "fast"], default="accurate")
    parser.add_argument("--language", default="auto", help="ASR language hint, for example zh, en, or auto")
    parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default="auto")
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--output-dir", type=Path, help="Optional explicit run directory")
    parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds")
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--ai-style", default="")
    parser.add_argument("--json", action="store_true", help="Print the underlying JSON result")
    args = parser.parse_args(argv)

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
        "--json",
    ]
    if args.asr_model:
        command.extend(["--asr-model", args.asr_model])
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
    except json.JSONDecodeError:
        print(completed.stdout, end="")
        return 0

    output_dir = Path(payload["output_dir"]).resolve()
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
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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


if __name__ == "__main__":
    raise SystemExit(main())
