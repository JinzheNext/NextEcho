from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .doctor import print_human_report, report_to_dict, run_doctor
from .speaker_transcript import build_speaker_transcript
from .transcription import DEFAULT_MODEL_NAME, transcribe_media_sources

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "outputs" / "transcriptions"
QUALITY_TO_MODEL = {
    "accurate": DEFAULT_MODEL_NAME,
    "fast": "base",
}


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def cmd_doctor(args: argparse.Namespace) -> int:
    report = run_doctor()
    if args.json:
        print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
    else:
        print_human_report(report)
    return 0 if report.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    command = [sys.executable, str(ROOT / "app.py")]
    env = None
    if args.host or args.port:
        # app.py currently owns host/port; keep flags accepted for future compatibility and simple UX.
        print("Note: serve currently uses app.py default http://127.0.0.1:8765")
    return subprocess.call(command, cwd=ROOT, env=env)


def cmd_transcribe(args: argparse.Namespace) -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    model = QUALITY_TO_MODEL.get(args.quality, DEFAULT_MODEL_NAME)
    payload = transcribe_media_sources(
        args.sources,
        output_dir=output_dir,
        model_name=model,
        language="auto",
        max_seconds=0,
    )
    result = {
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "results": payload["results"],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {payload['manifest_path']}")
        for item in payload["results"]:
            print(f"- {item.get('source_label')}: {item.get('status')} {item.get('text_path')}")
    return 0


def cmd_speaker_transcript(args: argparse.Namespace) -> int:
    payload = build_speaker_transcript(
        args.input_path,
        quality=args.quality,
        output_dir=args.output_dir,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {Path(payload['run_dir']) / 'transcript.speakers.md'}")
        print(payload["preview_text"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local transcription workbench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check local dependencies, models, and recommended quality")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    doctor_parser.set_defaults(func=cmd_doctor)

    serve_parser = subparsers.add_parser("serve", help="Start the local web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=cmd_serve)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe local files or media URLs locally")
    transcribe_parser.add_argument("sources", nargs="+", help="Local file paths or remote media/page URLs")
    transcribe_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    transcribe_parser.add_argument("--output-dir", type=Path)
    transcribe_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    transcribe_parser.set_defaults(func=cmd_transcribe)

    speaker_parser = subparsers.add_parser("speaker-transcript", help="Build a speaker-attributed interview transcript")
    speaker_parser.add_argument("input_path", help="Audio file, artifact directory, or single-source run directory")
    speaker_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    speaker_parser.add_argument("--output-dir", type=Path)
    speaker_parser.add_argument("--min-speakers", type=int, default=2)
    speaker_parser.add_argument("--max-speakers", type=int, default=2)
    speaker_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    speaker_parser.set_defaults(func=cmd_speaker_transcript)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
