from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .doctor import print_human_report, report_to_dict, run_doctor
from .config import NextEchoConfig, load_config
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


def config_for_args(args: argparse.Namespace) -> NextEchoConfig:
    return load_config(getattr(args, "config_path", None))


def _config_value(config: NextEchoConfig, *path: str, default: Any) -> Any:
    value = config.get_path(*path, default=default)
    return default if value is None else value


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    # ``SUPPRESS`` on subparsers preserves a global --config value supplied
    # before the command while still accepting --config after a subcommand.
    parser.add_argument("--config", dest="config_path", default=argparse.SUPPRESS, help="Path to an external TOML override")


def warn_legacy_whisper_model(args: argparse.Namespace) -> None:
    if getattr(args, "model", None):
        print("提示：--model 仍用于 Whisper 兼容路径；Qwen3-ASR 请使用 --asr-model。", file=sys.stderr)


def cmd_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(config_for_args(args))
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
    config = config_for_args(args)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor(config)
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
        config=config,
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "off"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
        backend=getattr(args, "speaker_backend", "auto"),
        config=config,
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
    config = config_for_args(args)
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
    doctor_report = run_doctor(config)
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
        config=config,
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "auto"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
        backend=getattr(args, "speaker_backend", "auto"),
        config=config,
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
    config = config_for_args(args)
    descriptors = resolve_feed(args.url, limit=args.limit)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or (RUNS_ROOT / f"run_{now_slug()}_{uuid4().hex[:8]}")
    doctor_report = run_doctor(config)
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
        config=config,
    )
    speaker_payload = maybe_build_speaker_transcript(
        output_dir,
        mode=getattr(args, "speaker_mode", "off"),
        min_speakers=getattr(args, "min_speakers", 2),
        max_speakers=getattr(args, "max_speakers", 6),
        backend=getattr(args, "speaker_backend", "auto"),
        config=config,
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
    config = config_for_args(args)
    doctor_report = run_doctor(config)
    payload = _call_speaker_builder(
        args.input_path,
        quality=args.quality,
        output_dir=args.output_dir,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        model_name=resolve_requested_model(args.model, args.quality, memory_gb=doctor_report.memory_gb),
        backend=getattr(args, "backend", "auto"),
        config=config,
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
    backend: str = "auto",
    config: NextEchoConfig | None = None,
) -> dict[str, Any] | None:
    if mode == "off":
        return None
    try:
        return _call_speaker_builder(
            run_dir,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend=backend,
            config=config,
        )
    except Exception as exc:
        if mode == "required":
            raise
        print(f"提示：说话人区分未完成，保留原始转录：{exc}", file=sys.stderr)
        return {"status": "failed", "error": str(exc), "run_dir": str(run_dir)}


def _call_speaker_builder(input_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Call the existing speaker module while tolerating older signatures.

    The diarization implementation is intentionally owned by the speaker
    pipeline.  CLI can expose ``--backend`` now and pass it whenever that
    pipeline supports it, without taking ownership of its internals.
    """

    call_kwargs = dict(kwargs)
    builder = build_speaker_transcript
    try:
        parameters = inspect.signature(builder).parameters
        call_kwargs = {key: value for key, value in call_kwargs.items() if key in parameters}
    except (TypeError, ValueError):
        # Builtins/proxies may not expose a signature; older implementations
        # accept the established keyword set, so retry only if backend/config
        # are rejected below.
        pass
    try:
        return builder(input_path, **call_kwargs)
    except TypeError as exc:
        if "backend" not in kwargs and "config" not in kwargs:
            raise
        fallback_kwargs = {key: value for key, value in call_kwargs.items() if key not in {"backend", "config"}}
        try:
            return builder(input_path, **fallback_kwargs)
        except TypeError:
            raise exc


def _optional_module(*module_names: str) -> Any | None:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _invoke_optional(module: Any, function_names: tuple[str, ...], *, kwargs: dict[str, Any], argv: list[str] | None = None) -> Any:
    for function_name in function_names:
        function = getattr(module, function_name, None)
        if not callable(function):
            continue
        if function_name == "main" and argv is not None:
            try:
                return function(argv)
            except TypeError:
                return function()
        try:
            parameters = inspect.signature(function).parameters
            accepted = {key: value for key, value in kwargs.items() if key in parameters}
        except (TypeError, ValueError):
            accepted = kwargs
        return function(**accepted)
    raise AttributeError(f"module {module.__name__!r} has no supported entry point")


def _print_optional_payload(payload: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif payload is not None:
        print(payload)


def cmd_speaker_enroll(args: argparse.Namespace) -> int:
    module = _optional_module("workbench.speaker_profiles", "workbench.speaker_enrollment")
    if module is None:
        message = "speaker enrollment is unavailable: install or provide workbench.speaker_profiles"
        if args.json:
            print(json.dumps({"status": "unavailable", "error": message}, ensure_ascii=False, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    config = config_for_args(args)
    try:
        payload = _invoke_optional(
            module,
            ("enroll_speaker", "enroll_profile", "speaker_enroll"),
            kwargs={
                "display_name": args.display_name,
                "name": args.display_name,
                "sample_paths": [str(path.expanduser().resolve()) for path in args.samples],
                "samples": [str(path.expanduser().resolve()) for path in args.samples],
                "consent": bool(args.consent),
                "backend": getattr(args, "backend", "auto"),
                "config": config,
            },
        )
    except AttributeError as exc:
        message = f"speaker enrollment module has no supported entry point: {exc}"
        print(message, file=sys.stderr)
        return 2
    except (FileNotFoundError, OSError, ValueError) as exc:
        message = f"speaker enrollment failed: {exc}"
        if args.json:
            print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    _print_optional_payload(payload, json_output=args.json)
    return 0


def cmd_speaker_profiles(args: argparse.Namespace) -> int:
    module = _optional_module("workbench.speaker_profiles", "workbench.speaker_enrollment")
    if module is None:
        message = "speaker profiles are unavailable: install or provide workbench.speaker_profiles"
        if args.json:
            print(json.dumps({"status": "unavailable", "error": message}, ensure_ascii=False, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    config = config_for_args(args)
    if args.profiles_action == "list":
        names = ("list_profiles", "get_profiles", "profiles_list")
        kwargs = {"config": config}
    else:
        names = ("remove_profile", "delete_profile", "profiles_remove")
        kwargs = {"person_id": args.person_id, "profile_id": args.person_id, "config": config}
    try:
        payload = _invoke_optional(module, names, kwargs=kwargs)
    except AttributeError as exc:
        message = f"speaker profile module has no supported entry point: {exc}"
        print(message, file=sys.stderr)
        return 2
    except (FileNotFoundError, OSError, ValueError) as exc:
        message = f"speaker profile operation failed: {exc}"
        if args.json:
            print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    _print_optional_payload(payload, json_output=args.json)
    return 0


def cmd_benchmark_speakers(args: argparse.Namespace) -> int:
    module = _optional_module("workbench.speaker_benchmark")
    if module is None:
        message = "speaker benchmark is unavailable: provide workbench.speaker_benchmark"
        if args.json:
            print(json.dumps({"status": "unavailable", "error": message}, ensure_ascii=False, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    config = config_for_args(args)
    kwargs = {
        "gold_dir": args.gold_dir.expanduser().resolve() if args.gold_dir else None,
        "config": config,
        "backend": args.backend,
        "json_output": bool(args.json),
    }
    argv = ["speakers"]
    if args.gold_dir:
        argv.extend(["--gold-dir", str(args.gold_dir)])
    if args.backend:
        argv.extend(["--backend", args.backend])
    if args.json:
        argv.append("--json")
    try:
        payload = _invoke_optional(module, ("run_speaker_benchmark", "benchmark_speakers", "main"), kwargs=kwargs, argv=argv)
    except AttributeError as exc:
        message = f"speaker benchmark module has no supported entry point: {exc}"
        print(message, file=sys.stderr)
        return 2
    _print_optional_payload(payload, json_output=args.json)
    return int(payload) if isinstance(payload, int) else 0


def cmd_list_models(args: argparse.Namespace) -> int:
    report = run_doctor(config_for_args(args))
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
        print(f"Config source: {report.config_source or 'built-in defaults'}")
        print(f"Recommended model: {report.recommended_model}")
        print(f"Reason: {report.recommended_model_reason}")
        print("\nAvailable Whisper models:")
        for item in payload["models"]:
            print(f"- {item['name']} [{item['tier']}] suggested memory >= {item['min_memory_gb']}GB")
            print(f"  {item['summary']}")
        print(f"\nQwen3-ASR MLX available: {payload['local_ai']['qwen_mlx_available']}")
        print("ASR candidates:")
        for candidate in payload["local_ai"].get("asr_candidates", []):
            print(f"- {candidate['engine']}: {candidate['model']} (eligible={candidate.get('eligible')})")
        print("Speaker backends:")
        for candidate in report.speaker_transcript.get("candidates", []):
            print(f"- {candidate['backend']} (available={candidate.get('available')})")
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
    default_config = load_config()
    parser.add_argument("--config", dest="config_path", default=None, help="Path to an external TOML override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check local dependencies, models, and recommended quality")
    _add_config_option(doctor_parser)
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    doctor_parser.set_defaults(func=cmd_doctor)

    serve_parser = subparsers.add_parser("serve", help="Start the local web UI")
    _add_config_option(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=cmd_serve)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe local files or media URLs locally")
    _add_config_option(transcribe_parser)
    transcribe_parser.add_argument("sources", nargs="+", help="Local file paths or remote media/page URLs")
    transcribe_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    transcribe_parser.add_argument("--language", default=_config_value(default_config, "asr", "language", default="zh"), help="ASR language hint, for example zh, en, or auto")
    transcribe_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    transcribe_parser.add_argument("--output-dir", type=Path)
    transcribe_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds; useful for gold-set comparisons")
    transcribe_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    transcribe_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    transcribe_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    transcribe_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    transcribe_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default=_config_value(default_config, "asr", "engine", default="auto"))
    transcribe_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    transcribe_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default=_config_value(default_config, "postprocess", "profile", default="fast"))
    transcribe_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default=_config_value(default_config, "diarization", "default_mode", default="auto"), help="Generate a speaker-attributed transcript after ASR")
    transcribe_parser.add_argument("--speaker-backend", dest="speaker_backend", default=_config_value(default_config, "diarization", "backend", default="auto"), help="Diarization backend (auto, pyannote-community-1, sherpa-onnx, resemblyzer)")
    transcribe_parser.add_argument("--min-speakers", type=int, default=_config_value(default_config, "diarization", "min_speakers", default=2))
    transcribe_parser.add_argument("--max-speakers", type=int, default=_config_value(default_config, "diarization", "max_speakers", default=6))
    transcribe_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    transcribe_parser.set_defaults(func=cmd_transcribe)

    resolve_parser = subparsers.add_parser("resolve-sources", help="Resolve URLs or local files into source descriptors")
    _add_config_option(resolve_parser)
    resolve_parser.add_argument("sources", nargs="+", help="Local file paths or remote media/page URLs")
    resolve_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    resolve_parser.set_defaults(func=cmd_resolve_sources)

    page_parser = subparsers.add_parser("transcribe-page", help="Resolve one media page URL and transcribe it locally")
    _add_config_option(page_parser)
    page_parser.add_argument("url", help="Single platform page URL such as Bilibili, Xiaoyuzhou, or Apple Podcasts")
    page_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    page_parser.add_argument("--language", default=_config_value(default_config, "asr", "language", default="zh"), help="ASR language hint, for example zh, en, or auto")
    page_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    page_parser.add_argument("--output-dir", type=Path)
    page_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds; useful for gold-set comparisons")
    page_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    page_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    page_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    page_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    page_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default=_config_value(default_config, "asr", "engine", default="auto"))
    page_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    page_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default=_config_value(default_config, "postprocess", "profile", default="fast"))
    page_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default=_config_value(default_config, "diarization", "default_mode", default="auto"), help="Generate a speaker-attributed transcript after ASR")
    page_parser.add_argument("--speaker-backend", dest="speaker_backend", default=_config_value(default_config, "diarization", "backend", default="auto"), help="Diarization backend")
    page_parser.add_argument("--min-speakers", type=int, default=_config_value(default_config, "diarization", "min_speakers", default=2))
    page_parser.add_argument("--max-speakers", type=int, default=_config_value(default_config, "diarization", "max_speakers", default=6))
    page_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    page_parser.set_defaults(func=cmd_transcribe_page)

    feed_parser = subparsers.add_parser("transcribe-feed", help="Resolve an RSS feed and transcribe recent episodes locally")
    _add_config_option(feed_parser)
    feed_parser.add_argument("url", help="RSS or podcast feed URL")
    feed_parser.add_argument("--limit", type=int, default=3)
    feed_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    feed_parser.add_argument("--language", default=_config_value(default_config, "asr", "language", default="zh"), help="ASR language hint, for example zh, en, or auto")
    feed_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    feed_parser.add_argument("--output-dir", type=Path)
    feed_parser.add_argument("--max-seconds", type=int, default=0, help="Only process the first N seconds per episode")
    feed_parser.add_argument("--glossary", type=Path, help="Optional glossary file with `wrong => right` rules")
    feed_parser.add_argument("--enhance", action="store_true", help="Generate enhanced transcript artifacts")
    feed_parser.add_argument("--ai-polish", action="store_true", help="Run optional AI polish on the enhanced transcript when configured")
    feed_parser.add_argument("--ai-style", default="", help="Optional style hint passed to AI polish")
    feed_parser.add_argument("--asr-engine", choices=["auto", "qwen3-asr", "whisper-cpp"], default=_config_value(default_config, "asr", "engine", default="auto"))
    feed_parser.add_argument("--asr-model", default="", help="Explicit Qwen3-ASR model ID")
    feed_parser.add_argument("--postprocess-profile", choices=["off", "balanced", "quality", "fast"], default=_config_value(default_config, "postprocess", "profile", default="fast"))
    feed_parser.add_argument("--speaker-mode", choices=["off", "auto", "required"], default=_config_value(default_config, "diarization", "default_mode", default="auto"), help="Generate a speaker-attributed transcript after ASR")
    feed_parser.add_argument("--speaker-backend", dest="speaker_backend", default=_config_value(default_config, "diarization", "backend", default="auto"), help="Diarization backend")
    feed_parser.add_argument("--min-speakers", type=int, default=_config_value(default_config, "diarization", "min_speakers", default=2))
    feed_parser.add_argument("--max-speakers", type=int, default=_config_value(default_config, "diarization", "max_speakers", default=6))
    feed_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    feed_parser.set_defaults(func=cmd_transcribe_feed)

    speaker_parser = subparsers.add_parser("speaker-transcript", help="Build a speaker-attributed interview transcript")
    _add_config_option(speaker_parser)
    speaker_parser.add_argument("input_path", help="Audio file, artifact directory, or single-source run directory")
    speaker_parser.add_argument("--quality", choices=sorted(QUALITY_TO_MODEL), default="accurate")
    speaker_parser.add_argument("--model", help="Explicit whisper.cpp model name or local model path")
    speaker_parser.add_argument("--output-dir", type=Path)
    speaker_parser.add_argument("--min-speakers", type=int, default=_config_value(default_config, "diarization", "min_speakers", default=2))
    speaker_parser.add_argument("--max-speakers", type=int, default=_config_value(default_config, "diarization", "max_speakers", default=6))
    speaker_parser.add_argument("--backend", default=_config_value(default_config, "diarization", "backend", default="auto"), help="Diarization backend")
    speaker_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    speaker_parser.set_defaults(func=cmd_speaker_transcript)

    list_models_parser = subparsers.add_parser("list-models", help="Show supported whisper.cpp models and the recommended choice for this machine")
    _add_config_option(list_models_parser)
    list_models_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    list_models_parser.set_defaults(func=cmd_list_models)

    download_model_parser = subparsers.add_parser("download-model", help="Download or reuse a specific whisper.cpp model locally")
    _add_config_option(download_model_parser)
    download_model_parser.add_argument("model", help="Model name such as tiny, base, small, medium, or large-v3-turbo-q5_0")
    download_model_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    download_model_parser.set_defaults(func=cmd_download_model)

    enhance_parser = subparsers.add_parser("enhance", help="Generate local polished transcript and knowledge artifacts for an existing run")
    _add_config_option(enhance_parser)
    enhance_parser.add_argument("run_dir", type=Path)
    enhance_parser.add_argument("--profile", choices=["balanced", "quality", "fast"], default="fast")
    enhance_parser.add_argument("--glossary", type=Path)
    enhance_parser.add_argument("--ai-style", default="")
    enhance_parser.add_argument("--json", action="store_true")
    enhance_parser.set_defaults(func=cmd_enhance)

    enroll_parser = subparsers.add_parser("speaker-enroll", help="Register local speaker samples through the optional speaker profile module")
    _add_config_option(enroll_parser)
    enroll_parser.add_argument("display_name", help="Display name for the local speaker profile")
    enroll_parser.add_argument("samples", nargs="+", type=Path, help="One or more local WAV/audio enrollment samples")
    enroll_parser.add_argument("--backend", default=_config_value(default_config, "speaker_identity", "embedding_backend", default="auto"), help="Embedding backend (auto or resemblyzer)")
    enroll_parser.add_argument("--consent", action="store_true", help="Record that the speaker consented to local enrollment")
    enroll_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    enroll_parser.set_defaults(func=cmd_speaker_enroll)

    profiles_parser = subparsers.add_parser("speaker-profiles", help="List or remove local speaker profiles")
    _add_config_option(profiles_parser)
    profiles_subparsers = profiles_parser.add_subparsers(dest="profiles_action", required=True)
    profiles_list_parser = profiles_subparsers.add_parser("list", help="List local speaker profiles")
    _add_config_option(profiles_list_parser)
    profiles_list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    profiles_list_parser.set_defaults(func=cmd_speaker_profiles)
    profiles_remove_parser = profiles_subparsers.add_parser("remove", help="Remove one local speaker profile")
    _add_config_option(profiles_remove_parser)
    profiles_remove_parser.add_argument("person_id")
    profiles_remove_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    profiles_remove_parser.set_defaults(func=cmd_speaker_profiles)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run optional local benchmark suites")
    _add_config_option(benchmark_parser)
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_action", required=True)
    speakers_benchmark_parser = benchmark_subparsers.add_parser("speakers", help="Evaluate speaker diarization against a local gold set")
    _add_config_option(speakers_benchmark_parser)
    speakers_benchmark_parser.add_argument("--gold-dir", type=Path, default=Path("benchmarks/speakers/local-private"))
    speakers_benchmark_parser.add_argument("--backend", default="auto")
    speakers_benchmark_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    speakers_benchmark_parser.set_defaults(func=cmd_benchmark_speakers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(raw_args)
    config = config_for_args(args)
    _apply_config_defaults(args, config, raw_args)
    return int(args.func(args))


def _apply_config_defaults(args: argparse.Namespace, config: NextEchoConfig, raw_args: list[str]) -> None:
    """Apply a --config file after parsing, even when it appears after a subcommand."""

    def provided(*names: str) -> bool:
        return any(name in raw_args for name in names)

    if hasattr(args, "language") and not provided("--language"):
        args.language = str(_config_value(config, "asr", "language", default=args.language))
    if hasattr(args, "asr_engine") and not provided("--asr-engine"):
        args.asr_engine = str(_config_value(config, "asr", "engine", default=args.asr_engine))
    if hasattr(args, "postprocess_profile") and not provided("--postprocess-profile"):
        args.postprocess_profile = str(_config_value(config, "postprocess", "profile", default=args.postprocess_profile))
    if hasattr(args, "speaker_mode") and not provided("--speaker-mode"):
        args.speaker_mode = str(_config_value(config, "diarization", "default_mode", default=args.speaker_mode))
    if hasattr(args, "speaker_backend") and not provided("--speaker-backend"):
        args.speaker_backend = str(_config_value(config, "diarization", "backend", default=args.speaker_backend))
    if hasattr(args, "backend") and not provided("--backend"):
        backend_path = ("speaker_identity", "embedding_backend") if getattr(args, "command", "") == "speaker-enroll" else ("diarization", "backend")
        args.backend = str(_config_value(config, *backend_path, default=args.backend))
    if hasattr(args, "min_speakers") and not provided("--min-speakers"):
        args.min_speakers = int(_config_value(config, "diarization", "min_speakers", default=args.min_speakers))
    if hasattr(args, "max_speakers") and not provided("--max-speakers"):
        args.max_speakers = int(_config_value(config, "diarization", "max_speakers", default=args.max_speakers))
    if getattr(args, "command", "") == "enhance" and not provided("--profile"):
        args.profile = str(_config_value(config, "postprocess", "profile", default=args.profile))


def _payload_exit_code(payload: dict[str, object]) -> int:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return 0
    return 1 if any(isinstance(item, dict) and item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
