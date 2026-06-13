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
from .sources import resolve_feed, resolve_sources
from .transcription import (
    DEFAULT_MODEL_NAME,
    list_supported_whisper_models,
    resolve_requested_model,
    resolve_whisper_model,
    transcribe_media_sources,
)

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
    doctor_report = run_doctor()
    model = resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb)
    descriptors = resolve_sources(args.sources)
    payload = transcribe_media_sources(
        descriptors,
        output_dir=output_dir,
        model_name=model,
        language="auto",
        max_seconds=0,
    )
    result = {
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {payload['manifest_path']}")
        for item in payload["results"]:
            print(f"- {item.get('source_label')}: {item.get('status')} {item.get('text_path')}")
        if payload.get("errors_path"):
            print(f"Errors: {payload['errors_path']}")
    return _payload_exit_code(payload)


def cmd_resolve_sources(args: argparse.Namespace) -> int:
    payload = [item.to_dict() for item in resolve_sources(args.sources)]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in payload:
            print(f"{item['platform']}: {item['title'] or item['canonical_url']} [{item['resolver']}]")
            if item.get("error"):
                print(f"  error: {item['error']}")
    return 0


def cmd_transcribe_page(args: argparse.Namespace) -> int:
    descriptors = resolve_sources([args.url])
    descriptor = descriptors[0]
    if descriptor.error:
        result = {"source": descriptor.to_dict(), "error": descriptor.error}
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(descriptor.error)
        return 1
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor()
    model = resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb)
    payload = transcribe_media_sources(
        [descriptor],
        output_dir=output_dir,
        model_name=model,
        language="auto",
        max_seconds=0,
    )
    result = {
        "source": descriptor.to_dict(),
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {payload['manifest_path']}")
        for item in payload["results"]:
            print(f"- {item.get('source_label')}: {item.get('status')} {item.get('text_path')}")
        if payload.get("errors_path"):
            print(f"Errors: {payload['errors_path']}")
    return _payload_exit_code(payload)


def cmd_transcribe_feed(args: argparse.Namespace) -> int:
    descriptors = resolve_feed(args.url, limit=args.limit)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor()
    model = resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb)
    payload = transcribe_media_sources(
        descriptors,
        output_dir=output_dir,
        model_name=model,
        language="auto",
        max_seconds=0,
    )
    result = {
        "feed_url": args.url,
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {payload['manifest_path']}")
        for item in payload["results"]:
            print(f"- {item.get('source_label')}: {item.get('status')} {item.get('text_path')}")
        if payload.get("errors_path"):
            print(f"Errors: {payload['errors_path']}")
    return _payload_exit_code(payload)


def cmd_speaker_transcript(args: argparse.Namespace) -> int:
    doctor_report = run_doctor()
    payload = build_speaker_transcript(
        args.input_path,
        quality=args.quality,
        output_dir=args.output_dir,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        model_name=resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {Path(payload['run_dir']) / 'transcript.speakers.md'}")
        print(payload["preview_text"])
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    report = run_doctor()
    payload = {
        "recommended_quality": report.recommended_quality,
        "recommended_quality_reason": report.recommendation_reason,
        "recommended_model": report.recommended_model,
        "recommended_model_reason": report.recommended_model_reason,
        "models": list_supported_whisper_models(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Recommended model: {report.recommended_model}")
        print(f"Reason: {report.recommended_model_reason}")
        print("\nAvailable Whisper models:")
        for item in payload["models"]:
            print(f"- {item['name']} [{item['tier']}] suggested memory >= {item['min_memory_gb']}GB")
            print(f"  {item['summary']}")
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    model_path = resolve_whisper_model(args.model)
    payload = {"model": args.model, "path": str(model_path)}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Downloaded or reused model: {args.model}")
        print(f"Path: {model_path}")
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
    transcribe_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    transcribe_parser.add_argument("--output-dir", type=Path)
    transcribe_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    transcribe_parser.set_defaults(func=cmd_transcribe)

    resolve_parser = subparsers.add_parser("resolve-sources", help="Resolve URLs or local files into source descriptors")
    resolve_parser.add_argument("sources", nargs="+", help="Local file paths or remote media/page URLs")
    resolve_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    resolve_parser.set_defaults(func=cmd_resolve_sources)

    page_parser = subparsers.add_parser("transcribe-page", help="Resolve one media page URL and transcribe it locally")
    page_parser.add_argument("url", help="Single platform page URL such as YouTube, Bilibili, or Xiaoyuzhou")
    page_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    page_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    page_parser.add_argument("--output-dir", type=Path)
    page_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    page_parser.set_defaults(func=cmd_transcribe_page)

    feed_parser = subparsers.add_parser("transcribe-feed", help="Resolve an RSS feed and transcribe recent episodes locally")
    feed_parser.add_argument("url", help="RSS or podcast feed URL")
    feed_parser.add_argument("--limit", type=int, default=3)
    feed_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    feed_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    feed_parser.add_argument("--output-dir", type=Path)
    feed_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    feed_parser.set_defaults(func=cmd_transcribe_feed)

    speaker_parser = subparsers.add_parser("speaker-transcript", help="Build a speaker-attributed interview transcript")
    speaker_parser.add_argument("input_path", help="Audio file, artifact directory, or single-source run directory")
    speaker_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    speaker_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    speaker_parser.add_argument("--output-dir", type=Path)
    speaker_parser.add_argument("--min-speakers", type=int, default=2)
    speaker_parser.add_argument("--max-speakers", type=int, default=2)
    speaker_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    speaker_parser.set_defaults(func=cmd_speaker_transcript)

    list_models_parser = subparsers.add_parser("list-models", help="Show supported whisper.cpp models and the recommended choice for this machine")
    list_models_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    list_models_parser.set_defaults(func=cmd_list_models)

    download_model_parser = subparsers.add_parser("download-model", help="Download or reuse a specific whisper.cpp model locally")
    download_model_parser.add_argument("model", help="Model name such as tiny, base, small, medium, or large-v3-turbo-q5_0")
    download_model_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    download_model_parser.set_defaults(func=cmd_download_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _payload_exit_code(payload: dict[str, object]) -> int:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return 0
    return 1 if any(isinstance(item, dict) and item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
