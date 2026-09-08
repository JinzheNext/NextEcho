from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .doctor import print_human_report, report_to_dict, run_doctor
from .enhancement import load_glossary_text
from .postprocess import process_transcript_artifacts
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


def warn_legacy_whisper_model(args: argparse.Namespace) -> None:
    if getattr(args, "model", None):
        print("提示：--model 仍用于 Whisper 兼容路径；Qwen3-ASR 请使用 --asr-model。", file=sys.stderr)


def cmd_doctor(args: argparse.Namespace) -> int:
    report = run_doctor()
    if args.json:
        print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
    else:
        print_human_report(report)
    return 0 if report.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from app import RUNS_ROOT as APP_RUNS_ROOT, app as web_app

    APP_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    web_app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    warn_legacy_whisper_model(args)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor()
    model = resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb)
    descriptors = resolve_sources(args.sources)
    payload = transcribe_media_sources(
        descriptors,
        output_dir=output_dir,
        model_name=model,
        language=args.language,
        max_seconds=int(getattr(args, "max_seconds", 0) or 0),
        glossary_text=load_glossary_text(args.glossary) if getattr(args, "glossary", None) else "",
        enable_enhancement=bool(args.enhance or args.glossary or args.ai_polish or getattr(args, "postprocess_profile", "off") != "off"),
        ai_polish=bool(args.ai_polish),
        ai_style=args.ai_style or "",
        asr_engine=getattr(args, "asr_engine", "auto"),
        asr_model=getattr(args, "asr_model", "") or "",
        postprocess_profile=getattr(args, "postprocess_profile", "off"),
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "off"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
    )
    result = {
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
        "speaker_transcript": speaker_payload,
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
    warn_legacy_whisper_model(args)
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
        language=args.language,
        max_seconds=int(getattr(args, "max_seconds", 0) or 0),
        glossary_text=load_glossary_text(args.glossary) if getattr(args, "glossary", None) else "",
        enable_enhancement=bool(args.enhance or args.glossary or args.ai_polish or getattr(args, "postprocess_profile", "off") != "off"),
        ai_polish=bool(args.ai_polish),
        ai_style=args.ai_style or "",
        asr_engine=getattr(args, "asr_engine", "auto"),
        asr_model=getattr(args, "asr_model", "") or "",
        postprocess_profile=getattr(args, "postprocess_profile", "off"),
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "auto"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
    )
    result = {
        "source": descriptor.to_dict(),
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
        "speaker_transcript": speaker_payload,
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
    warn_legacy_whisper_model(args)
    descriptors = resolve_feed(args.url, limit=args.limit)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor()
    model = resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb)
    payload = transcribe_media_sources(
        descriptors,
        output_dir=output_dir,
        model_name=model,
        language=args.language,
        max_seconds=int(getattr(args, "max_seconds", 0) or 0),
        glossary_text=load_glossary_text(args.glossary) if getattr(args, "glossary", None) else "",
        enable_enhancement=bool(args.enhance or args.glossary or args.ai_polish or getattr(args, "postprocess_profile", "off") != "off"),
        ai_polish=bool(args.ai_polish),
        ai_style=args.ai_style or "",
        asr_engine=getattr(args, "asr_engine", "auto"),
        asr_model=getattr(args, "asr_model", "") or "",
        postprocess_profile=getattr(args, "postprocess_profile", "off"),
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "off"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
    )
    result = {
        "feed_url": args.url,
        "output_dir": str(payload["output_dir"]),
        "manifest_path": str(payload["manifest_path"]),
        "errors_path": str(payload["errors_path"]) if payload.get("errors_path") else "",
        "results": payload["results"],
        "speaker_transcript": speaker_payload,
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


def maybe_build_speaker_transcript(
    run_dir: Path,
    *,
    mode: str,
    min_speakers: int,
    max_speakers: int,
) -> dict[str, Any] | None:
    if mode == "off":
        return None
    try:
        return build_speaker_transcript(
            run_dir,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as exc:
        if mode == "required":
            raise
        print(f"提示：说话人区分未完成，保留原始转录：{exc}", file=sys.stderr)
        return {"status": "failed", "error": str(exc), "run_dir": str(run_dir)}


def cmd_list_models(args: argparse.Namespace) -> int:
    report = run_doctor()
    payload = {
        "recommended_quality": report.recommended_quality,
        "recommended_quality_reason": report.recommendation_reason,
        "recommended_model": report.recommended_model,
        "recommended_model_reason": report.recommended_model_reason,
        "models": list_supported_whisper_models(),
        "local_ai": report.local_ai,
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
        print(f"\nQwen3-ASR MLX available: {payload['local_ai']['qwen_mlx_available']}")
        print(f"Ollama ready: {payload['local_ai']['ollama'].get('ready')}")
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


def cmd_enhance(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    item_dirs = [run_dir] if (run_dir / "transcript.json").exists() else sorted((run_dir / "items").glob("*"))
    processed_dirs: list[Path] = []
    results: list[dict] = []
    for item_dir in item_dirs:
        if not (item_dir / "transcript.json").exists():
            continue
        results.append(
            process_transcript_artifacts(
                item_dir,
                profile=args.profile,
                user_glossary_text=load_glossary_text(args.glossary) if args.glossary else "",
                style=args.ai_style or "",
            )
        )
        processed_dirs.append(item_dir)
        metadata_path = item_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["enhancement"] = results[-1]
            metadata.update(results[-1].get("paths") or {})
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not results:
        raise FileNotFoundError("No transcript.json found in the supplied run directory")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item_dir, audit in zip(processed_dirs, results):
            for item in manifest.get("results") or []:
                if Path(str(item.get("output_dir") or "")).resolve() == item_dir.resolve():
                    item["enhancement"] = audit
                    item.update(audit.get("paths") or {})
        manifest.setdefault("postprocess", {})["profile"] = args.profile
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps({"run_dir": str(run_dir), "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"Generated knowledge artifacts for {len(results)} item(s).")
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
    transcribe_parser.add_argument("--language", default="auto", help="ASR language hint, for example zh, en, or auto")
    transcribe_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    transcribe_parser.add_argument("--output-dir", type=Path)
    transcribe_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds; useful for gold-set comparisons")
    transcribe_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    transcribe_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    transcribe_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    transcribe_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    transcribe_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default="auto")
    transcribe_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    transcribe_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default="fast")
    transcribe_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default="off", help="Generate a speaker-attributed transcript after ASR")
    transcribe_parser.add_argument("--min-speakers", type=int, default=2)
    transcribe_parser.add_argument("--max-speakers", type=int, default=6)
    transcribe_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    transcribe_parser.set_defaults(func=cmd_transcribe)

    resolve_parser = subparsers.add_parser("resolve-sources", help="Resolve URLs or local files into source descriptors")
    resolve_parser.add_argument("sources", nargs="+", help="Local file paths or remote media/page URLs")
    resolve_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    resolve_parser.set_defaults(func=cmd_resolve_sources)

    page_parser = subparsers.add_parser("transcribe-page", help="Resolve one media page URL and transcribe it locally")
    page_parser.add_argument("url", help="Single platform page URL such as Bilibili, Xiaoyuzhou, or Apple Podcasts")
    page_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    page_parser.add_argument("--language", default="auto", help="ASR language hint, for example zh, en, or auto")
    page_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    page_parser.add_argument("--output-dir", type=Path)
    page_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds; useful for gold-set comparisons")
    page_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    page_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    page_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    page_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    page_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default="auto")
    page_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    page_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default="fast")
    page_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default="auto", help="Generate a speaker-attributed transcript after ASR")
    page_parser.add_argument("--min-speakers", type=int, default=2)
    page_parser.add_argument("--max-speakers", type=int, default=6)
    page_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    page_parser.set_defaults(func=cmd_transcribe_page)

    feed_parser = subparsers.add_parser("transcribe-feed", help="Resolve an RSS feed and transcribe recent episodes locally")
    feed_parser.add_argument("url", help="RSS or podcast feed URL")
    feed_parser.add_argument("--limit", type=int, default=3)
    feed_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    feed_parser.add_argument("--language", default="auto", help="ASR language hint, for example zh, en, or auto")
    feed_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    feed_parser.add_argument("--output-dir", type=Path)
    feed_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds per episode")
    feed_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    feed_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    feed_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    feed_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    feed_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default="auto")
    feed_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    feed_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default="fast")
    feed_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default="off", help="Generate a speaker-attributed transcript after ASR")
    feed_parser.add_argument("--min-speakers", type=int, default=2)
    feed_parser.add_argument("--max-speakers", type=int, default=6)
    feed_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    feed_parser.set_defaults(func=cmd_transcribe_feed)

    speaker_parser = subparsers.add_parser("speaker-transcript", help="Build a speaker-attributed interview transcript")
    speaker_parser.add_argument("input_path", help="Audio file, artifact directory, or single-source run directory")
    speaker_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    speaker_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    speaker_parser.add_argument("--output-dir", type=Path)
    speaker_parser.add_argument("--min-speakers", type=int, default=2)
    speaker_parser.add_argument("--max-speakers", type=int, default=6)
    speaker_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    speaker_parser.set_defaults(func=cmd_speaker_transcript)

    list_models_parser = subparsers.add_parser("list-models", help="Show supported whisper.cpp models and the recommended choice for this machine")
    list_models_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    list_models_parser.set_defaults(func=cmd_list_models)

    download_model_parser = subparsers.add_parser("download-model", help="Download or reuse a specific whisper.cpp model locally")
    download_model_parser.add_argument("model", help="Model name such as tiny, base, small, medium, or large-v3-turbo-q5_0")
    download_model_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    download_model_parser.set_defaults(func=cmd_download_model)

    enhance_parser = subparsers.add_parser("enhance", help="Generate local polished transcript and knowledge artifacts for an existing run")
    enhance_parser.add_argument("run_dir", type=Path)
    enhance_parser.add_argument("--profile", choices=["balanced", "quality", "fast"], default="fast")
    enhance_parser.add_argument("--glossary", type=Path)
    enhance_parser.add_argument("--ai-style", default="")
    enhance_parser.add_argument("--json", action="store_true")
    enhance_parser.set_defaults(func=cmd_enhance)

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
